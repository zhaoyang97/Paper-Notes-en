---
title: >-
  [Paper Note] SAVA-X: Ego-to-Exo Imitation Error Detection via Scene-Adaptive View Alignment and Bidirectional Cross View Fusion
description: >-
  [CVPR 2026][Video Understanding][cross-view imitation error detection] This paper proposes SAVA-X, a framework comprising three complementary modules—adaptive sampling, scene-aware view embedding, and bidirectional cross-attention fusion—to address cross-view temporal error detection in the exocentric-demonstration-to-egocentric-imitation setting, achieving comprehensive improvements over existing baselines on the EgoMe benchmark.
tags:
  - CVPR 2026
  - Video Understanding
  - cross-view imitation error detection
  - adaptive sampling
  - scene-aware view embedding
  - bidirectional cross-attention fusion
  - egocentric-exocentric video
date: 2026-05-08
content_hash: c9119d432d28865a
---

# SAVA-X: Ego-to-Exo Imitation Error Detection via Scene-Adaptive View Alignment and Bidirectional Cross View Fusion

**Conference**: CVPR 2026
**arXiv**: [2603.12764](https://arxiv.org/abs/2603.12764)
**Code**: [jack1ee/SAVAX](https://github.com/jack1ee/SAVAX)
**Area**: Video Understanding
**Keywords**: cross-view imitation error detection, adaptive sampling, scene-aware view embedding, bidirectional cross-attention fusion, egocentric-exocentric video

## TL;DR

This paper proposes SAVA-X, a framework comprising three complementary modules—adaptive sampling, scene-aware view embedding, and bidirectional cross-attention fusion—to address cross-view temporal error detection in the exocentric-demonstration-to-egocentric-imitation setting, achieving comprehensive improvements over existing baselines on the EgoMe benchmark.

## Background & Motivation

**Strong practical demand**: In industrial assembly, medical training, and robot imitation learning, operators (first-person/ego) must execute actions based on third-person (exo) demonstrations, making error detection critical for quality control.

**Prior methods limited to single-view**: Existing error detection approaches (e.g., PREGO) assume single-view input and cannot handle the realistic scenario where demonstrations and executions come from different viewpoints.

**Temporal misalignment**: Ego/Exo videos are recorded asynchronously with different durations and pacing; direct feature alignment causes false positives, as duration differences alone do not constitute errors.

**Severe redundancy interference**: Long videos contain substantial irrelevant content that dilutes attention mechanisms and increases false positives; experiments show that baseline methods degrade in performance as the number of input frames increases.

**Significant viewpoint domain gap**: The ego view focuses on local hand-object interactions while the exo view captures whole-body posture and scene layout; their appearance and motion statistics differ substantially, making direct feature fusion unreliable.

**Lack of unified evaluation protocol**: The task had not been formally defined, and the absence of standardized baselines and evaluation frameworks has impeded research progress.

## Method

### Overall Architecture: Align–Fuse–Detect

SAVA-X employs a frozen video encoder (TSP, pretrained on ActivityNet) to extract per-frame features, which are then processed sequentially through three core modules and a deformable Transformer encoder-decoder for final prediction:

1. **Adaptive Sampling (AS)** → redundancy removal + temporal alignment
2. **Scene-aware View-dictionary Embedding (SVE)** → reducing the cross-view domain gap
3. **Bidirectional Cross-attention Fusion (BiX)** → complementary evidence aggregation
4. Decoder outputs egocentric temporal segments with imitation correctness judgments

### Key Design 1: Gated Adaptive Sampling (AS)

- **Exo side**: Computes saliency scores via self-attention + FFN; retains keyframes via Gumbel Top-K hard selection.
- **Ego side**: Scores frames via cross-attention with already-sampled Exo features as Key/Value, making Ego sampling sensitive to demonstration keypoints.
- **Residual gating**: Introduces soft gating alongside hard selection via $\mathbf{g} = \mathbf{1} + \alpha(\text{Norm}(\mathbf{s}) - \mathbf{1})$ to provide stable gradients to the scorer.
- **Regularization**: Selection entropy $\mathcal{L}_{\text{sel}}$ prevents selection collapse; VICReg-style $\mathcal{L}_{\text{vic}}$ prevents dimensional collinearity.

### Key Design 2: Scene-Aware View-Dictionary Embedding (SVE)

- Maintains a shared view-scene dictionary $\mathbf{D} \in \mathbb{R}^{M \times d}$, whose row vectors capture common view sub-factors (e.g., "close-range hand-object interaction," "whole-body motion structure").
- Each view stream retrieves adaptive view embeddings from the dictionary via temperature-scaled cross-attention: $\mathbf{VE}^u = \text{CrossAttn}(\hat{\mathbf{Z}}^u / \tau, \mathbf{D})$.
- **Two-stage injection**: Injected once into each Ego/Exo stream before fusion, and again at multiple temporal levels within the encoder.
- **Attention entropy regularization** $\mathcal{L}_{\text{view-ent}}$: Prevents overly sharp attention distributions and encourages uniform dictionary coverage.
- **Dictionary diversity regularization** $\mathcal{L}_{\text{dict-div}}$: Enforces approximate orthogonality among normalized dictionary rows.

### Key Design 3: Bidirectional Cross-Attention Fusion (BiX)

- Symmetric bidirectional cross-attention: Ego→Exo and Exo→Ego computed in parallel.
- **Learnable gated residual mixing**: $\mathbf{F}^{ego} = (1-\boldsymbol{\gamma}^e)\tilde{\mathbf{Z}}^{ego} + \boldsymbol{\gamma}^e \mathbf{E}^\star$, with gate values generated by sigmoid over concatenated features.
- Final fusion: $\tilde{\mathbf{Z}}^{fused} = \frac{1}{2}(\mathbf{F}^{ego} + \mathbf{F}^{exo})$.
- The Exo→Ego direction provides boundary and step-ordering cues; the Ego→Exo direction contributes hand-object details and local causal information.

### Loss & Training

- DVC loss $\mathcal{L}_{\text{DVC}}$ (Hungarian set prediction, following PDVC configuration)
- Imitation discrimination loss $\mathcal{L}_{\text{Imit}}$ (weight $\lambda_{\text{Imit}}=0.5$)
- Regularization terms: $\mathcal{L}_{\text{sel}}$, $\mathcal{L}_{\text{vic}}$, $\mathcal{L}_{\text{view-ent}}$, $\mathcal{L}_{\text{dict-div}}$ (weights 0.01–0.05)

## Key Experimental Results

### Dataset & Setup

- **EgoMe** dataset: 7,902 asynchronous Exo-Ego video pairs (~82.8 hours); train/val/test = 4,777/997/2,128
- Feature extraction: TSP (frozen), feature dimension $d=512$
- Optimizer: AdamW, learning rate 1e-4, batch size 16

### Main Results (Table 1)

| Method | Val AUPRC@0.3 | Val AUPRC@0.5 | Val AUPRC@0.7 | Val Mean | Val tIoU |
|------|:---:|:---:|:---:|:---:|:---:|
| PDVC | 28.21 | 20.48 | 7.95 | 18.88 | 58.58 |
| Exo2EgoDVC | 31.33 | 20.27 | 7.49 | 19.69 | 59.06 |
| ActionFormer | 31.37 | 15.41 | 2.63 | 16.47 | 48.89 |
| TriDet | 30.04 | 14.61 | 2.44 | 15.70 | 49.05 |
| PDVC (Ego-only) | 19.35 | 13.91 | 5.11 | 12.79 | 57.63 |
| **SAVA-X** | **33.56** | **24.04** | **9.48** | **22.36** | **59.31** |

SAVA-X achieves a Mean AUPRC improvement of +2.67 (+13.56%) over the strongest baseline Exo2EgoDVC, attaining top performance at all thresholds.

### Ablation Study (Table 2)

| AS | SVE | BiX | Mean AUPRC | tIoU |
|:---:|:---:|:---:|:---:|:---:|
| | | | 18.88 | 58.58 |
| ✓ | | | 20.90 | 58.88 |
| | ✓ | | 21.29 | 59.27 |
| | | ✓ | 21.06 | 58.27 |
| ✓ | ✓ | | 21.82 | 58.96 |
| ✓ | | ✓ | 20.32 | 58.14 |
| | ✓ | ✓ | 22.33 | 58.76 |
| ✓ | ✓ | ✓ | **22.36** | **59.31** |

### Key Findings

- **Three modules are complementary**: AS, SVE, and BiX individually yield gains of +10.7%, +12.8%, and +11.6%, respectively, with their combination achieving optimal performance.
- **SVE+BiX is the strongest pairwise combination**: Among two-module combinations, SVE+BiX performs best, indicating that domain gap reduction combined with bidirectional verification is most critical.
- **AS+BiX is relatively weak**: Direct fusion without view adaptation is susceptible to domain shift and noise.
- **Single-view input degrades substantially**: Ego-only PDVC drops to a Mean AUPRC of 12.79, confirming the necessity of exocentric demonstration information.
- **Adaptive sampling is more effective at higher frame rates**: Greater redundancy at high frame rates means retaining a small number of high-scoring frames suffices to improve performance.
- **SVE outperforms fixed view embeddings**: Fixed learnable tokens yield limited gains, whereas the adaptive dictionary covers cross-scene variation.
- **Exo→Ego direction is more critical**: Unidirectional ablations show that Exo→Ego alone approaches the performance of the full bidirectional design, as the task objective is error detection on the ego stream.

## Highlights & Insights

- The paper is the first to formally define the ego-to-exo imitation error detection task and establish a unified evaluation protocol.
- Each of the three modules addresses one core challenge (redundancy / domain gap / fusion) in an orthogonal and complementary manner.
- The scene-aware dictionary view embedding is a creative design that achieves cross-scene adaptability via a learnable dictionary.
- Gated adaptive sampling balances the efficiency of hard selection with the gradient stability of soft gating.
- The ablation and component analyses are highly thorough, covering frame rate, Top-K ratio, dictionary size, fusion variants, and domain gap visualization.

## Limitations & Future Work

- Validation is limited to the single EgoMe dataset; generalizability remains unknown.
- The frozen TSP feature extractor (pretrained on ActivityNet) may not be sufficiently adapted to egocentric video.
- Absolute performance remains low (Mean AUPRC of only 22.36), leaving a substantial gap to practical deployment.
- Large-scale video foundation models (e.g., InternVideo, VideoMAE v2) are not explored for feature extraction.
- Dictionary size and regularization weights require manual tuning, with no automatic selection mechanism proposed.
- Inference speed and computational cost are not discussed.

## Related Work & Insights

- **Temporal action localization**: TAL methods such as ActionFormer and TriDet perform poorly in the cross-view setting (Mean AUPRC only 14–16).
- **Dense video captioning**: PDVC serves as a strong base architecture; Exo2EgoDVC introduces view-invariant adversarial learning.
- **Ego-Exo transfer**: Ego-Exo (Li et al. 2021) investigates representation transfer from third-person to first-person perspectives.
- **Procedural error detection**: PREGO (single-view online error detection); Lee et al. 2024 (error-free prototype-based detection).
- **Adaptive frame selection**: Buch et al. 2025 propose flexible frame selection for efficient video inference.

## Rating

- Novelty: ⭐⭐⭐⭐ (novel task formulation, targeted three-module design)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (highly detailed ablations, comprehensive component analysis)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, motivation and methodology well articulated)
- Value: ⭐⭐⭐⭐ (clear practical application scenarios, though absolute performance is limited)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Decompose and Transfer: CoT-Prompting Enhanced Alignment for Open-Vocabulary Temporal Action Detection](decompose_and_transfer_cot-prompting_enhanced_alignment_for_open-vocabulary_temp.md)
- [\[CVPR 2026\] Seen-to-Scene: Keep the Seen, Generate the Unseen for Video Outpainting](seen_to_scene_keep_the_seen_generate_the_unseen_for_video_outpainting.md)
- [\[CVPR 2026\] CVA: Context-aware Video-text Alignment for Video Temporal Grounding](cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)
- [\[CVPR 2026\] CLCR: Cross-Level Semantic Collaborative Representation for Multimodal Learning](clcr_cross-level_semantic_collaborative_representation_for_multimodal_learning.md)
- [\[CVPR 2026\] SVAgent: Storyline-Guided Long Video Understanding via Cross-Modal Multi-Agent Collaboration](svagent_storyline_guided_long_video_understanding_via_cross_modal_multi_agent_collaboration.md)

</div>

<!-- RELATED:END -->
