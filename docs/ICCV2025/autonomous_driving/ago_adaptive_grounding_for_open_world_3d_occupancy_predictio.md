---
title: >-
  [Paper Note] AGO: Adaptive Grounding for Open World 3D Occupancy Prediction
description: >-
  [ICCV 2025][Autonomous Driving][3D occupancy prediction] This paper proposes the AGO framework, which handles known categories via noise-augmented grounding training and unknown categories via a modality adapter for adap…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "3D occupancy prediction"
  - "open world"
  - "VLM knowledge distillation"
  - "self-supervised"
  - "adaptive alignment"
date: 2026-05-08
content_hash: edc909931c0be8c5
---

# AGO: Adaptive Grounding for Open World 3D Occupancy Prediction

**Conference**: ICCV 2025
**arXiv**: [2504.10117](https://arxiv.org/abs/2504.10117)
**Code**: [https://github.com/EdwardLeeLPZ/AGO](https://github.com/EdwardLeeLPZ/AGO)
**Area**: Autonomous Driving / 3D Occupancy Prediction / Open World
**Keywords**: 3D occupancy prediction, open world, VLM knowledge distillation, self-supervised, adaptive alignment

## TL;DR
This paper proposes the AGO framework, which handles known categories via noise-augmented grounding training and unknown categories via a modality adapter for adaptive alignment. An information entropy-based open-world recognizer dynamically selects the optimal features at inference time. AGO surpasses VEON by 4.09 mIoU on the Occ3D-nuScenes self-supervised benchmark while exhibiting open-world zero-shot/few-shot transfer capability.

## Background & Motivation

### Limitations of Prior Work

**Background**: 3D semantic occupancy prediction relies on costly 3D annotations. Self-supervised methods exploit VLMs to generate pseudo-labels, but face two core issues: (1) conventional supervision based on pseudo-labels is constrained to a fixed label space and cannot predict unknown categories; (2) directly aligning with VLM image embeddings introduces a severe modality gap (image–text similarity ranges only ~0.1), leading to unreliable predictions.

### Paper Goals

**Goal**: How can a 3D occupancy model maintain strong prediction performance on known categories while acquiring open-world generalization to predict unknown categories?

## Method

### Overall Architecture
Surround-view images → ResNet-101 for 2D feature extraction → TPVFormer for 3D voxel embedding construction → Dual-path training: (1) Grounding training: similarity computation between 3D embeddings and text/noise embeddings; (2) Adaptive alignment: modality adapter (MLP) maps 3D embeddings into the VLM image embedding space → At inference, an open-world recognizer selects the optimal prediction based on information entropy.

### Key Designs
1. **Noise-Augmented Grounding Training**: Rather than using a conventional classifier, dot products between 3D voxel features and text embeddings are directly computed as logits. Noise texts randomly sampled from a general vocabulary (~100 per step) serve as negative samples, enhancing discriminability for known categories. A learnable "free" embedding is introduced to represent free voxels.
2. **Adaptive Alignment via Modality Adapter**: A two-layer MLP (with softplus activation) maps 3D embeddings into a new space aligned with VLM image embeddings, avoiding the modality conflict that arises from imposing both text and image alignment losses on the same embedding. The cosine similarity loss is computed only over visible, non-empty voxels.
3. **Open-World Recognizer**: For each voxel, the prediction entropy of the original 3D embedding and the adapted 3D embedding are compared; the one with lower entropy is selected. Original embeddings are more reliable for known categories, while adapted embeddings are more reliable for unknown categories.

### Loss & Training
- $L_{\text{total}} = L_{\text{Grounding}} + L_{\text{Occ}} + L_{\text{Alignment}}$
- Grounding training: CE + Lovász-softmax loss
- Alignment: cosine similarity loss (visible non-empty voxels only)
- Pseudo-labels: Grounded SAM generates 2D masks → multi-frame aggregation + ray casting + semantic voting → 3D pseudo-labels
- AdamW, lr = 1e-3, 24 epochs, 8 × A100

## Key Experimental Results

### Occ3D-nuScenes Self-Supervised Benchmark

| Method | Parameters | mIoU↑ |
|--------|-----------|-------|
| SelfOcc | - | 9.30 |
| OccNeRF | - | 9.53 |
| GaussTR | VFMs | 11.70 |
| VEON | ViT-L | 15.14 |
| **AGO** | 62.5M | **19.23** |

mIoU gain: +4.09 vs. VEON, with only 9.2% of VEON's parameter count.

### Open-World Evaluation

| Stage | Method | Known mIoU | Unknown mIoU | Total mIoU |
|-------|--------|-----------|-------------|-----------|
| Pre-training | SelfOcc | 16.61 | 0.00 | 8.31 |
| Pre-training | POP-3D | 16.39 | 0.94 | 8.66 |
| Pre-training | **AGO** | **22.13** | **3.59** | **12.86** |
| Few-shot fine-tuning | **AGO** | **38.15** | **8.50** | **14.43** |

### Ablation Study
- Alignment only (Align): 10.28 mIoU → Grounding only (Gro.): 19.08 → AGO (grounding + adaptive alignment): 19.23
- Applying both grounding and alignment on the same embedding causes a drop (18.89): modality conflict
- Noise prompts: +0.26 mIoU; occupancy loss $L_{\text{Occ}}$: +0.96 mIoU
- Minimum entropy criterion outperforms maximum confidence criterion (3.6 vs. 3.1 unknown mIoU)
- ResNet-50 achieves 15.23 mIoU, still surpassing all prior methods

## Highlights & Insights
- **Grounding training as a replacement for conventional classifiers**: Using text embeddings directly as logits enables seamless label space switching between training and inference.
- **Noise text negatives are a clever design**: Randomly sampled vocabulary words serve as hard negatives, enhancing discriminability at nearly zero additional cost.
- **Modality adapter decouples text/image alignment**: This effectively avoids modality conflict and provides a general solution for handling text–image embedding inconsistencies in VLMs.
- **Entropy-based decision mechanism**: A simple yet effective approach for switching prediction sources between known and unknown categories.

## Limitations & Future Work
- Temporal information is not exploited (single-frame prediction).
- Text prompt design remains relatively simple (fine-grained sub-category decomposition unexplored).
- Limited improvement on certain rare dynamic categories (e.g., trailer).
- Zero-shot capability for unknown categories remains limited (unknown mIoU only 3.59 at the pre-training stage).

## Related Work & Insights
- **vs. VEON**: VEON integrates multiple large foundation models (ViT-L, etc., 678M parameters); AGO uses only 62.5M parameters yet achieves +4.09 mIoU.
- **vs. POP-3D**: POP-3D relies purely on alignment without geometric or semantic cues, yielding only 0.94 unknown mIoU.
- **vs. SelfOcc**: Pure self-supervision with a fixed label space; entirely incapable of predicting unknown categories (0.00 mIoU).

## Relevance to My Research
- Open-world 3D perception is a critical challenge in autonomous driving.
- The grounding training paradigm is transferable to other tasks requiring flexible label spaces.
- The modality adapter solution for VLM embedding inconsistency has broad applicability.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of grounding training and adaptive alignment is relatively novel; the open-world recognizer is concise and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Closed-set + three-stage open-world evaluation + multiple benchmarks + comprehensive ablations + Waymo validation.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear; ablations are presented in a progressive, well-structured manner.
- **Value**: ⭐⭐⭐⭐ The methodology for open-world occupancy prediction offers important reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SA-Occ: Satellite-Assisted 3D Occupancy Prediction in Real World](sa-occ_satellite-assisted_3d_occupancy_prediction_in_real_world.md)
- [\[ICCV 2025\] ALOcc: Adaptive Lifting-Based 3D Semantic Occupancy and Cost Volume-Based Flow Predictions](alocc_adaptive_liftingbased_3d_semantic_occupancy_and_cost_v.md)
- [\[ICCV 2025\] Semantic Causality-Aware Vision-Based 3D Occupancy Prediction](semantic_causality-aware_vision-based_3d_occupancy_prediction.md)
- [\[ICCV 2025\] EmbodiedOcc: Embodied 3D Occupancy Prediction for Vision-based Online Scene Understanding](embodiedocc_embodied_3d_occupancy_prediction_for_vision-based_online_scene_under.md)
- [\[ICCV 2025\] Towards Open-World Generation of Stereo Images and Unsupervised Matching](towards_open-world_generation_of_stereo_images_and_unsupervised_matching.md)

</div>

<!-- RELATED:END -->
