---
title: >-
  [Paper Note] Dynamic Reflections: Probing Video Representations with Text Alignment
description: >-
  [ICLR 2026][video representation alignment] This paper is the first to extend the Platonic Representation Hypothesis (PRH) from static image–text to the temporal video–text domain. Through systematic evaluation of 121 visual and language models, it reveals that increasing the number of frames and captions at test time can nearly double alignment scores, and proposes a saturating scaling law with $R^2 > 0.98$ to quantify this behavior.
tags:
  - ICLR 2026
  - video representation alignment
  - Platonic representation hypothesis
  - test-time scaling laws
  - cross-modal alignment
  - self-supervised learning
date: 2026-05-08
content_hash: e4a3477f9804827c
---

# Dynamic Reflections: Probing Video Representations with Text Alignment

**Conference**: ICLR 2026
**arXiv**: [2511.02767](https://arxiv.org/abs/2511.02767)
**Code**: [https://video-prh.github.io](https://video-prh.github.io)
**Area**: Interpretability / Representation Learning
**Keywords**: video representation alignment, Platonic representation hypothesis, test-time scaling laws, cross-modal alignment, self-supervised learning

## TL;DR

This paper is the first to extend the Platonic Representation Hypothesis (PRH) from static image–text to the temporal video–text domain. Through systematic evaluation of 121 visual and language models, it reveals that increasing the number of frames and captions at test time can nearly double alignment scores, and proposes a saturating scaling law with $R^2 > 0.98$ to quantify this behavior.

## Background & Motivation

The Platonic Representation Hypothesis (PRH) posits that as neural networks scale in capacity, data diversity, and task variety, the internal representations learned by different models converge toward a shared, modality-agnostic universal statistical model. Huh et al. (2024) previously validated this hypothesis in the static image–text setting, finding significant structural similarity between the latent spaces of independently trained visual encoders (e.g., DINOv2) and language encoders.

However, prior validation exhibits two fundamental limitations:

1. **Modality restriction**: All experiments were confined to static modalities (images and text). The motion, causal relationships, and temporal dependencies present in video data were entirely overlooked in representation alignment studies. Although PRH was formulated for all modalities, its validity in the temporal domain remained an open question.

2. **Interpretability of alignment scores**: Huh et al. (2024) raised an unresolved question — a maximum alignment score of 0.16 could not be judged as high or low in absolute terms.

The central observation of this paper is that previously reported limited alignment is largely attributable to **insufficient test-time data** (single frame + single caption). By providing multiple video frames and multiple textual descriptions, alignment scores can be substantially raised to nearly 0.4 without modifying any trained model. This finding establishes test-time scaling as a new dimension complementary to training-time scaling.

## Method

### Overall Architecture

This paper adopts the Mutual $k$-NN (MkNN) metric proposed by Huh et al. (2024) to measure cross-modal representation alignment. Given $N$ video–text pairs $\mathcal{S} = \{(v_1, c_1), \ldots, (v_N, c_N)\}$, embedding matrices $\mathbf{X} \in \mathbb{R}^{N \times p}$ and $\mathbf{Y} \in \mathbb{R}^{N \times q}$ are obtained via video and text encoders, respectively. Two $k$-nearest-neighbor binary indicator matrices $\mathbf{M_X}$ and $\mathbf{M_Y}$ are then constructed, and the alignment score is computed as:

$$\mathcal{A}^{\text{MkNN}}(\mathbf{X}, \mathbf{Y}) = \frac{1}{kN} \sum_{i=1}^{N} \sum_{j=1}^{N} (\mathbf{M_X} \odot \mathbf{M_Y})_{ij}$$

where $\odot$ denotes the Hadamard product and $k$ is typically set to 10 for a dataset of 1,024 samples. In addition, intermediate-layer combinations across both encoders are searched to select the layer pair that maximizes the alignment score.

The core extension of this work is to generalize the framework from the "single frame + single caption" setting to a "multi-frame video + multiple captions" setting, systematically investigating the effect of test-time data richness on alignment scores.

### Key Designs

**Multi-frame video encoding strategy**: For a video encoder natively processing $n_o$ frames, $n_f$ frames are sampled via uniform linear interpolation. When $n_f > n_o$, the video is partitioned into multiple sub-clips of length $n_o$, each encoded separately and averaged. When $n_f = 1$, the setting degenerates to the prior image–text alignment configuration. For image models, two variants are provided: single-frame only, and averaged features across 8 frames (image model adapted to video).

**Multi-caption text encoding strategy**: Multiple captions are concatenated into a single long string and encoded by a text encoder (including LLMs such as the Gemma 2 series); intermediate-layer features are extracted and averaged over the token dimension to yield features of shape $[\text{layer}, \text{hidden\_dim}]$. The VATEX dataset naturally supports multi-caption evaluation, providing 10 independently annotated captions per video. For the PVD dataset, which contains only a single long description per video, Gemini-2.5 Pro is used to decompose each description into 10 shorter captions.

**Saturating test-time scaling law**: Based on empirical observations, a parametric saturation model is proposed to quantify the joint dependence of alignment scores on frame count $n_f$ and caption count $n_c$:

$$\text{score}(n_f, n_c) = S_{\infty} - (C_f \cdot n_f^{-\alpha} + C_c \cdot n_c^{-\beta})$$

where $S_{\infty}$ is the theoretical saturating alignment score, $C_f$ and $C_c$ are error coefficients for frames and captions respectively, and $\alpha$ and $\beta$ are decay exponents. This model is analogous to the training-time compute-optimal scaling laws of Hoffmann et al. (2022): $S_{\infty}$ corresponds to the ideal alignment accuracy, and the subtracted terms represent the error penalty incurred by limited test-time data.

### Loss & Training

This paper is an analytical evaluation study and does not involve training new models. Its core methodology can be summarized as **test-time scaling**:

- **Visual-side scaling**: Incrementally increasing $n_f$ from 1 to 80 frames, leveraging more temporal information via sub-clip encoding and average pooling.
- **Text-side scaling**: Incrementally increasing $n_c$ from 1 to 10 captions, improving semantic coverage by concatenating multiple descriptions.
- **Layer search strategy**: Exhaustively searching all intermediate-layer combinations across both encoders, selecting the pair with the highest alignment score as the final result.

This paradigm complements training-time resource scaling (model parameter count, training data volume), demonstrating that test-time data refinement is also an effective means of improving representation alignment.

## Key Experimental Results

### Main Results: Video–Text Alignment Scores

Evaluated on VATEX (10-second videos + 10 annotations) and PVD datasets using a test set of 1,024 samples:

| Visual Model | Type | Text Encoder | Frames / Captions | MkNN Alignment Score |
|---|---|---|---|---|
| DINOv2 | Image (single frame) | Best non-Gemma | 1 frame / 1 caption | ~0.18 |
| DINOv2 | Image (single frame) | Gemma 2 9B-it | 1 frame / 1 caption | ~0.206 |
| DINOv2 | Image→Video (8-frame avg.) | Gemma 2 9B-it | 8 frames / 1 caption | ~0.223 |
| VideoMAEv2 | Native video | Gemma 2 9B-it | Multi-frame / multi-caption | ~0.41 ($S_{\infty}$) |
| DINOv2 | Image→Video | Gemma 2 9B-it | Multi-frame / multi-caption | ~0.37 ($S_{\infty}$) |

Core finding: From the minimal setting (0.18) to full exploitation of test-time data (0.41), the alignment score increases by more than **2×**.

### Ablation Study: Scaling Law Fitting and Analysis

| Fitted Parameter | VideoMAEv2 | DINOv2 | Interpretation |
|---|---|---|---|
| $S_{\infty}$ (saturation score) | 0.41 | 0.37 | Video model has a higher theoretical upper bound |
| $C_f$ (frame error coefficient) | 0.15 | 0.05 | Video model is **3×** more sensitive to frame count |
| $C_c$ (caption error coefficient) | 0.13 | 0.13 | Text-side influence is comparable |
| $\alpha$ (frame decay exponent) | 0.75 | 1.76 | Video model decays more slowly; requires more frames to saturate |
| $\beta$ (caption decay exponent) | 1.30 | 1.40 | Caption-side decay is similar |
| $R^2$ | 0.9791 | 0.9964 | Excellent fit quality |

| Ablation Dimension | Range | Key Observation |
|---|---|---|
| Frame count $n_f$ | 1 → 80 | Alignment increases steadily; video models benefit far more than image models |
| Caption count $n_c$ | 1 → 10 | Average alignment gain of **60%**; growth is fastest at small $n_c$ |
| Downstream semantic tasks (SSv2, K700) | — | **Strong positive correlation** with alignment score |
| Downstream non-semantic tasks (depth, pose) | — | Also positively correlated, except for point tracking |
| Temporal sensitivity (Test of Time) | $k=1,2,3$ | Near-perfect alignment at $k=3$; large gaps at $k=1,2$, suggesting LLMs favor bag-of-words |
| Temporal sensitivity (VideoComp) | Positive vs. negative captions | Higher-alignment models are more susceptible to temporal reordering |
| Synthesized multi-caption (PVD) | 1 → 10 synthetic captions | Decomposing a single long description into short captions also improves alignment |

## Highlights & Insights

- **First extension of PRH to the temporal domain**: A systematic evaluation covering 85 visual models × 36 language models fills the gap in representation alignment research for the video modality, demonstrating that temporal information provides strong signals for semantic understanding.
- **Discovery of test-time scaling laws**: Analogous to training-time compute-optimal scaling laws, this paper proposes test-time data scaling laws. The $R^2 > 0.98$ fit quality indicates that the dependence of alignment scores on frame and caption counts follows a highly predictable power-law behavior.
- **Resolution of a key open question**: The question posed by Huh et al. (2024) — "Is an alignment score of 0.16 high or low?" — receives a clear answer: it is an underestimate caused by limited test-time data; with sufficient data, scores exceeding 0.4 are achievable.
- **Practical utility of zero-shot evaluation metrics**: The strong correlation between video–text alignment and downstream task performance (both semantic and non-semantic) suggests that alignment scores can serve as a proxy for expensive task-specific evaluations in guiding video model development.
- **Potential of self-supervised video models**: VideoMAEv2, trained without any text supervision, surpasses DINOv2 in alignment score, demonstrating that purely video-based self-supervised training can yield representations highly aligned with the language space.

## Limitations & Future Work

1. **Insufficient coverage of local tasks**: The correlation between point tracking and alignment is weak, indicating that the MkNN metric emphasizes global semantics and struggles to capture fine-grained local spatiotemporal abilities.
2. **Performance gap for native video models**: Many native video models achieve lower alignment scores than image models with frame-averaged features, suggesting that the training paradigms for video encoders still have room for improvement.
3. **Representation utilization in generative video models**: The latent representations of current generative video models (e.g., video diffusion models) are poorly aligned with text; leveraging their understanding capabilities remains an open problem.
4. **Limited dataset diversity**: The study primarily uses VATEX (10-second short videos) and the PVD dataset, with insufficient coverage of long videos and complex temporal reasoning scenarios.
5. **Confounding factors in the caption effect**: Increasing the number of captions simultaneously increases both semantic coverage and viewpoint diversity; the independent contributions of these two factors to alignment have not been disentangled.

## Related Work & Insights

This paper sits at the intersection of three research directions: (1) *Platonic Representation Hypothesis and emergent alignment* — extending the static-modality work of Huh et al. (2024) and Maniparambil et al. (2024) to the temporal domain for the first time; (2) *Self-supervised video representation learning* — represented by large-scale unlabeled video pretraining methods such as VideoMAEv2 and V-JEPA, for which this paper provides a new zero-shot evaluation approach; and (3) *Scaling law research* — forming a dual with the training-time scaling laws of Hoffmann et al. (2022) and opening a systematic research direction for "test-time scaling." Furthermore, the finding that Gemma 2 series models — purely text-generative models — serve as the best-performing text encoders echoes the conclusions of Zhang et al. (2025) regarding the importance of language models in multimodal alignment.

## Rating

- Novelty: ⭐⭐⭐⭐ — First extension of PRH to the video domain; the test-time scaling law is novel and predictive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 121 model combinations with broad coverage, multi-dataset validation, and rigorous scaling law fitting.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rich and intuitive figures, precise articulation of core findings.
- Value: ⭐⭐⭐⭐ — Offers meaningful insights for both the evaluation paradigm of video representation learning and the theory of multimodal alignment.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Dynamic Reflections: Probing Video Representations with Text-Driven Reasoning](dynamic_reflections_probing_video_representations_with_text_driven_reasoning.md)
- [\[ICLR 2026\] One Language, Two Scripts: Probing Script-Invariance in LLM Concept Representations](one_language_two_scripts_probing_script-invariance_in_llm_concept_representation.md)
- [\[ICLR 2026\] Beyond Linear Probes: Dynamic Safety Monitoring for Language Models](beyond_linear_probes_dynamic_safety_monitoring_for_language_models.md)
- [\[ICLR 2026\] Uncovering Grounding IDs: How External Cues Shape Multimodal Binding](uncovering_grounding_ids_how_external_cues_shape_multimodal_binding.md)
- [\[ICLR 2026\] Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning](uni-ntfm_a_unified_foundation_model_for_eeg_signal_representation_learning.md)

<!-- RELATED:END -->
