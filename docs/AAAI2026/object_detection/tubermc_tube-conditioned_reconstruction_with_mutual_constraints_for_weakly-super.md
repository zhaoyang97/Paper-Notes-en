---
title: >-
  [Paper Note] TubeRMC: Tube-conditioned Reconstruction with Mutual Constraints for Weakly-supervised Spatio-Temporal Video Grounding
description: >-
  [AAAI 2026][Object Detection][Weakly-supervised spatio-temporal video grounding] This paper proposes TubeRMC, a framework that generates text-conditioned candidate tubes and performs tube-conditioned reconstruction along…
tags:
  - "AAAI 2026"
  - "Object Detection"
  - "Weakly-supervised spatio-temporal video grounding"
  - "tube reconstruction"
  - "vision-language alignment"
  - "mutual constraint learning"
  - "STVG"
date: 2026-05-08
content_hash: ea0f6c2ea244e7a6
---

# TubeRMC: Tube-conditioned Reconstruction with Mutual Constraints for Weakly-supervised Spatio-Temporal Video Grounding

**Conference**: AAAI 2026
**arXiv**: [2511.10241](https://arxiv.org/abs/2511.10241)  
**Code**: None  
**Area**: Object Detection
**Keywords**: Weakly-supervised spatio-temporal video grounding, tube reconstruction, vision-language alignment, mutual constraint learning, STVG

## TL;DR

This paper proposes TubeRMC, a framework that generates text-conditioned candidate tubes and performs tube-conditioned reconstruction along temporal, spatial, and spatio-temporal dimensions, augmented by spatial-temporal mutual constraints to improve weakly-supervised spatio-temporal video grounding.

## Background & Motivation

**Spatio-temporal video grounding (STVG)** aims to localize a spatio-temporal tube — a sequence of bounding boxes within a specified temporal interval — in an untrimmed video given a natural language query. This is a highly challenging task requiring complex vision-language understanding and spatio-temporal reasoning.

Existing fully-supervised methods (e.g., TubeDETR) rely on expensive tube-text annotations. To reduce annotation costs, weakly-supervised STVG (WSTVG) methods train using only video-text pairs, without bounding box or temporal annotations.

**Core limitations of existing weakly-supervised methods**:

**Late-fusion paradigm limitations**: Methods such as WINNER and VCMA follow a detect-then-match pipeline — generating tube proposals with unimodal detectors (e.g., Faster-RCNN) and subsequently matching them with text. Since tube generation is entirely independent of the text description, two critical issues arise:
   - **Target identification failure**: The detector cannot perceive the targets described in the text.
   - **Tracking inconsistency**: Cross-frame target identification is unstable.

**Infeasibility of directly concatenating frame-level results**: Although pretrained visual grounding models (e.g., MDETR) can capture text-conditioned object localization, naively concatenating frame-level results to form a spatio-temporal tube is ineffective due to inconsistent cross-frame target identification and lack of spatio-temporal understanding.

**Insufficiency of existing reconstruction methods**: Existing weakly-supervised video temporal grounding methods (e.g., CNM, Kim2024) focus solely on temporal reconstruction, neglecting the spatial-textual correspondence.

**Core insight**: A tube that matches the target event should be capable of correctly reconstructing masked key phrases in the query sentence. As illustrated in Figure 1(b), a correct tube can reconstruct keywords such as "white man" and "sits down."

## Method

### Overall Architecture

The TubeRMC framework operates at three levels:

1. **Text-conditioned tube generation**: A pretrained visual grounding model (MDETR) is used to extract frame-level cross-modal representations and spatial localization results.
2. **Tube-conditioned reconstruction learning**: Reconstruction is performed along temporal, spatial, and spatio-temporal dimensions to comprehensively capture tube-text correspondences.
3. **Mutual constraint learning**: Bidirectional temporal-spatial constraints are introduced to enhance proposal quality.

### Key Designs

#### 1. **Model Architecture**

**Static cross-modal extraction**: A pretrained MDETR (ResNet-101 + RoBERTa-base) extracts cross-modal representations and localization results per frame. For each frame, all predicted boxes are ranked by the confidence score of the subject token, and the highest-scoring box is selected as the frame-level prediction. These are concatenated to form a bounding box tube $B \in \mathbb{R}^{T \times 4}$ and a confidence vector $S \in \mathbb{R}^{T \times 1}$.

**Spatio-temporal modeling**: Cross-modal features are fed into TimeSFormer to obtain cross-modal features $F_t \in \mathbb{R}^{T \times (H \times W + L) \times d}$ and global frame features $F_g \in \mathbb{R}^{T \times d}$, which are subsequently passed to:
- **Spatial Boxes Refiner**: Models inter-frame contextual relationships via cross-attention to predict offsets that refine MDETR outputs.
- **Temporal Proposals Generator**: Uses $K$ learnable queries within a temporal decoder to model temporal context.
- **Spatio-Temporal Decoder**: Integrates spatial and temporal information for event-level prediction.

#### 2. **Tube-conditioned Reconstruction Learning (Core Contribution)**

Unlike methods that rely solely on temporal reconstruction, this paper proposes a **Tube-conditioned Reconstructor (TR)** that takes both temporal and spatial masks as input.

**Gaussian mask representation**: To enable backpropagation, temporal intervals, spatial boxes, and spatio-temporal tubes are converted into Gaussian distributions:
- 1D Gaussian → temporal attention mask $M_t \in \mathbb{R}^T$
- 2D Gaussian → spatial attention mask $M_b \in \mathbb{R}^{HW}$
- 3D Gaussian → spatio-temporal mask $T \times HW$

**TR architecture**: Comprises a Tube-conditioned Encoder (6 layers) and a Masked-text Reconstructor:
- The encoder contains a local branch (focusing on local spatio-temporal context) and a global branch (supplementing global temporal information).
- Mask-Attention mechanism: $M\text{-}att(Q,K,V,M) = (\text{softmax}(\frac{QK^T}{\sqrt{d}}) \bigotimes M) V$, guiding the encoder to attend to visual regions corresponding to the mask.

**Three reconstruction strategies**:
- **Temporal reconstruction**: Masks predicates and related nouns (motion information) using a 1D Gaussian temporal mask.
- **Spatial reconstruction**: Masks subject nouns and adjectives (appearance information) using a 2D Gaussian spatial mask.
- **Spatio-temporal reconstruction**: Randomly masks tokens (with higher probability for verbs, nouns, and adjectives) using a 3D Gaussian mask.

#### 3. **Mutual Constraint Learning**

**Space-to-time constraint**: The confidence scores of spatial boxes guide temporal proposal generation. Top-$K$ high-scoring frames are selected as temporal centers for positive proposals, while the lowest-scoring frames serve as negative proposals. The loss minimizes the overlap between positive and negative proposals.

**Time-to-space constraint**: Ensures spatial continuity of targets across adjacent frames within the same scene. For predicted boxes in adjacent frames within each temporal proposal, cases where IoU falls below a threshold are penalized.

### Loss & Training

Total loss: $L_{total} = L_{rec} + L_{ipc} + L_{ivc} + L_{mc}$

- **Reconstruction loss** $L_{rec}$: Sum of cross-entropy losses from the three reconstruction strategies.
- **Inter-proposal contrastive loss** $L_{ipc}$: Ensures the reconstruction loss of positive proposals is lower than that of negative proposals (with margin).
- **Intra-video contrastive loss** $L_{ivc}$: Spatio-temporal reconstruction with positive samples versus hard/easy negatives (inverted temporal mask, uniform mask).
- **Mutual constraint loss** $L_{mc}$: Space-to-time and time-to-space constraints combined.

Training settings: $K=4$ (HCSTVG/VidSTG), 3-layer transformer for spatio-temporal modeling, 6 layers for TR, margin parameters $\beta_1=0.5$, $\beta_2=0.7$, $\beta_3=0.5$, $\beta_4=0.7$.

## Key Experimental Results

### Main Results

| Dataset | Metric | TubeRMC | Prev. SOTA (VCMA) | Gain |
|--------|------|---------|---------------|------|
| HCSTVG-v1 | m_vIoU | **19.38** | 14.64 | +4.74 |
| HCSTVG-v1 | vIoU@0.3 | **23.88** | 18.60 | +5.28 |
| HCSTVG-v2 | m_vIoU | **20.64** | — | — |
| VidSTG-Decl | m_vIoU | **15.93** | 14.45 | +1.48 |
| VidSTG-Decl | vIoU@0.3 | **25.16** | 18.57 | +6.59 |
| VidSTG-Inter | m_vIoU | **13.47** | 13.25 | +0.22 |

Compared to MDETR baselines (HCSTVG-v2): TubeRMC outperforms MDETR-Zero by 8.43% and MDETR+CPL by 5.55%.

### Ablation Study

**Reconstruction strategy ablation** (HCSTVG-v1):

| Spatial | Temporal | Spatio-Temp. | m_vIoU | vIoU@0.3 | Note |
|------|------|------|--------|----------|------|
| ✗ | ✗ | ✗ | 14.91 | 14.87 | No reconstruction baseline |
| ✓ | ✗ | ✗ | 15.24 | 14.74 | +Spatial |
| ✗ | ✓ | ✗ | 17.59 | 20.25 | +Temporal (+2.68, significant) |
| ✓ | ✓ | ✗ | 18.12 | 20.43 | Spatial+Temporal complementary |
| ✓ | ✓ | ✓ | **19.38** | **23.88** | All strategies, best |

**Mutual constraint ablation**:

| s-to-t | t-to-s | m_vIoU | m_tIoU | m_sIoU |
|--------|--------|--------|--------|--------|
| ✗ | ✗ | 15.87 | 26.69 | 59.83 |
| ✓ | ✗ | 17.65 | 29.04 | 59.95 |
| ✗ | ✓ | 17.07 | 27.38 | 60.13 |
| ✓ | ✓ | **19.38** | **30.94** | **61.67** |

**Visual grounding model substitution**: Replacing MDETR with G-DINO (Swin-B) raises m_vIoU to 21.15%; however, MDETR is used by default to balance speed and performance.

### Key Findings

1. Temporal reconstruction yields the largest single-component gain (+2.68 m_vIoU), underscoring the importance of temporal modeling in WSTVG.
2. The three reconstruction strategies are complementary, and their combination achieves the best performance.
3. The space-to-time constraint primarily improves m_tIoU (+2.35/+3.56), while the time-to-space constraint primarily improves m_sIoU.
4. Stronger visual grounding models (e.g., G-DINO-Swin-B) can further boost performance.

## Highlights & Insights

1. **Three-dimensional reconstruction paradigm**: This work is the first to learn tube-text correspondences simultaneously across 1D/2D/3D dimensions, yielding a clean and elegant formulation.
2. **Mutual constraint mechanism**: Spatial information guides temporal proposal generation, while temporal proposals enforce spatial continuity, forming a virtuous cycle.
3. **Plug-and-play visual grounding model**: The framework is agnostic to the choice of visual grounding model and can benefit from future model advances.
4. **Reconstruction as understanding**: Evaluating tube-text matching quality by assessing whether masked text can be reconstructed constitutes an elegant weakly-supervised training signal.

## Limitations & Future Work

1. Under severe viewpoint changes and occlusions, MDETR may incorrectly assign bounding boxes to other individuals performing similar actions (e.g., the third row in the visualization case).
2. Performance remains dependent on the quality of the visual grounding model; when the pretraining corpus of MDETR diverges significantly from the target dataset (e.g., VidSTG Interrogative), performance is limited.
3. Tracking algorithms could be incorporated to generate higher-quality tube proposals.
4. Automatic learning of 3D Gaussian mask parameters is worth exploring.

## Related Work & Insights

- The reconstruction paradigm is inspired by weakly-supervised video temporal grounding (WVTG); this work extends it from 1D to 2D and 3D.
- Visual grounding models such as MDETR and G-DINO provide stronger initialization for weakly-supervised approaches.
- The mutual constraint idea is generalizable to other tasks requiring spatial-temporal co-reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐ — The three-dimensional reconstruction and mutual constraint mechanisms are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive ablations across four datasets and multiple settings.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with complete mathematical formulations.
- Value: ⭐⭐⭐⭐ — Advances the state of the art in weakly-supervised STVG research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection](temporal_object-aware_vision_transformer_for_few-shot_video_object_detection.md)
- [\[ICLR 2026\] SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection](../../ICLR2026/object_detection/spwood_sparse_partial_weakly-supervised_oriented_object_detection.md)
- [\[ICLR 2026\] Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting (WS-COC)](../../ICLR2026/object_detection/bootstrapping_mllm_for_weakly-supervised_class-agnostic_object_counting.md)
- [\[NeurIPS 2025\] Spatio-Temporal Graphs Beyond Grids: Benchmark for Maritime Anomaly Detection](../../NeurIPS2025/object_detection/spatio-temporal_graphs_beyond_grids_benchmark_for_maritime_anomaly_detection.md)
- [\[ICCV 2025\] Sim-DETR: Unlock DETR for Temporal Sentence Grounding](../../ICCV2025/object_detection/sim-detr_unlock_detr_for_temporal_sentence_grounding.md)

</div>

<!-- RELATED:END -->
