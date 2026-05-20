---
title: >-
  [Paper Note] Large-scale Pre-training for Grounded Video Caption Generation
description: >-
  [ICCV 2025][Object Detection][grounded video caption generation] This paper proposes the GROVE model along with a large-scale automatic annotation pipeline…
tags:
  - "ICCV 2025"
  - "Object Detection"
  - "grounded video caption generation"
  - "large-scale pre-training"
  - "spatio-temporal grounding"
  - "automatic annotation"
  - "bounding box prediction"
date: 2026-05-08
content_hash: 13f8946f2db53d6b
---

# Large-scale Pre-training for Grounded Video Caption Generation

**Conference**: ICCV 2025
**arXiv**: [2503.10781](https://arxiv.org/abs/2503.10781)  
**Code**: [Project Page](https://ekazakos.github.io/grounded_video_caption_generation/)  
**Area**: Object Detection
**Keywords**: grounded video caption generation, large-scale pre-training, spatio-temporal grounding, automatic annotation, bounding box prediction

## TL;DR

This paper proposes the GROVE model along with a large-scale automatic annotation pipeline, constructing the HowToGround1M pre-training dataset (1M videos) and the manually annotated iGround dataset (3,513 videos). GROVE jointly performs video caption generation and multi-object spatio-temporal bounding box localization, achieving state-of-the-art results on iGround, VidSTG, ActivityNet-Entities, and other benchmarks.

## Background & Motivation

Grounded Video Caption Generation requires simultaneously addressing two challenging tasks: (1) generating natural language descriptions at the video level, and (2) predicting temporally dense and consistent bounding boxes for noun phrases in the generated captions. Compared to grounded captioning in images, the video setting introduces additional difficulties: objects may disappear due to occlusion, and bounding box predictions must remain temporally consistent across frames.

The key bottleneck of this problem lies in the **lack of large-scale datasets**:

- **Small dataset scale**: VidSTG contains 36.2K instances, HC-STVG contains 10.1K, and manual annotation costs are prohibitively high.
- **Incomplete annotations**: Many datasets localize only a single spatio-temporal tube per short text description, making multi-object grounding infeasible.
- **Poor temporal consistency**: Datasets such as ActivityNet-Entities annotate bounding boxes for only one frame per segment, resulting in temporally sparse coverage.
- **Domain constraints**: Some datasets are restricted to specific scenarios (e.g., egocentric video).

Furthermore, directly applying image-based grounded captioning models to video suffers from inter-frame inconsistency — frame-by-frame models produce temporally incoherent noun phrase annotations.

## Method

### Overall Architecture

The GROVE system consists of two components:

1. **Large-scale automatic annotation pipeline** (3 stages): constructs the HowToGround1M pre-training data from the HowTo100M dataset.
2. **GROVE model**: a grounded video caption generation model extending GLaMM to the video domain, incorporating spatio-temporal adapters, a bounding box decoder, and a temporal objectness head.

### Key Designs

1. **Three-Stage Automatic Annotation Method**:

    - **Function**: Aggregates frame-level grounded captions into video-level, temporally consistent, dense annotations.
    - **Mechanism**:
        - **Stage 1 (Frame-level grounded captioning)**: Applies the GLaMM image grounded captioning model frame-by-frame to obtain per-frame text descriptions and bounding boxes (converting segmentation masks to bounding boxes).
        - **Stage 2 (Video-level caption aggregation)**: Extracts SVO (subject-verb-object) triples from frame-level captions and uses Llama-2 with in-context learning to aggregate them into a video-level caption, with key noun phrases identified.
        - **Stage 3 (Temporally consistent annotation)**: Matches frame-level noun phrases to video-level noun phrases via LLM-based text classification, ensuring consistent labels for the same object across frames and forming per-object spatio-temporal tracks.
    - **Design Motivation**: Processing frames independently with an image model leads to temporal inconsistency (e.g., the same cup labeled as "cup," "mug," and "glass" in different frames). The three-stage approach leverages the semantic understanding of LLMs to unify annotations.

2. **Spatio-temporal Adapters**:

    - **Function**: Trainable spatio-temporal adapters inserted between frozen image encoder layers to endow the image-based backbone with video temporal modeling capability.
    - **Mechanism**: $a(o) = o + \tanh(\alpha) \times f(o)$, where $o$ is the output of the preceding encoder layer, $\alpha$ is a trainable scalar initialized to 0, and $f(\cdot)$ is the adapter layer. At the start of training, $\tanh(0)=0$ effectively suppresses the adapter output, allowing the network to gradually adjust its contribution.
    - **Design Motivation**: End-to-end fine-tuning of the entire video encoder is computationally expensive and may destroy pretrained image representations. The residual connection with zero-initialized trainable parameters enables stable training while preserving pretrained knowledge.

3. **Bounding Box Decoder**:

    - **Function**: Adapts a pretrained mask decoder into a bounding box decoder that predicts per-frame bounding boxes for each detection token.
    - **Mechanism**: Detection token embeddings serve as queries, while visual features from the Grounding Video Encoder serve as keys and values. Although $\mathcal{V}_g(\cdot)$ operates on the full video, cross-attention is applied per frame to predict per-frame object locations: $p_{bb} = h_{bb}(o_d) \in \mathbb{R}^{T \times N_d \times 4}$.
    - **Design Motivation**: Reusing large-scale pretrained decoder weights (e.g., from SAM) and simplifying mask prediction to bounding box prediction — bounding box annotations are cheaper to obtain and sufficiently accurate for compact object localization.

4. **Temporal Objectness Head**:

    - **Function**: Explicitly predicts whether each object is visible in each frame (i.e., whether it is occluded or has left the scene).
    - **Mechanism**: $p_{tobj} = h_{tobj}(o_d) \in \mathbb{R}^{T \times N_d \times 1}$. At inference, a threshold is applied to suppress bounding box predictions for frames where the object is deemed invisible.
    - **Design Motivation**: Objects frequently disappear and reappear in video, which is a core challenge. Unlike objectness in image detection (determining whether a region contains an object), temporal objectness determines "whether the object is visible in this frame," addressing the false positive predictions that arise when conventional methods are forced to predict bounding boxes for occluded frames.

### Loss & Training

- Standard caption generation loss (language model cross-entropy)
- Bounding box regression loss (L1 + GIoU)
- Temporal objectness loss (binary cross-entropy)
- During pre-training, the visual backbone and LLM are frozen; the LLM's embedding and output layers, adapters, and decoder are trained.
- The model is first pre-trained at scale on HowToGround1M, then fine-tuned on the smaller, high-quality iGround dataset.

## Key Experimental Results

### Main Results

| Dataset | Metric | GROVE PT+FT | Prev. SOTA | Gain |
|--------|------|------------|----------|------|
| iGround (Center) | METEOR | 21.4 | 11.9 (GLaMM) | +9.5 |
| iGround (Center) | CIDEr | 83.5 | 29.9 (GLaMM) | +53.6 |
| iGround (Center) | AP50 | 31.7 | 20.8 (GLaMM) | +10.9 |
| iGround (All) | AP50 | 40.0 | 27.1 (Auto-annotation) | +12.9 |
| iGround (All) | Recall | 28.7 | 20.4 (Auto-annotation) | +8.3 |
| VidSTG (declarative) | msIoU | **63.7** | 61.9 (DenseVOC) | +1.8 |
| VidSTG (interrogative) | msIoU | **55.5** (FT) | 39.7 (VideoGLaMM) | +15.8 |
| ActivityNet-Entities | F1_loc_per_sent | **77.29** | 59.20 (GVD) | +18.09 |

### Ablation Study

| Configuration | METEOR | CIDEr | AP50 | Recall | Notes |
|------|--------|-------|------|--------|------|
| No adapter, no unfreeze | 19.2 | 82.2 | 36.8 | 25.9 | Lowest baseline |
| Adapter, no unfreeze | 19.7 | 88.9 | 39.2 | 26.4 | Adapter contributes clearly |
| Adapter + unfreeze decoder | **19.7** | **92.6** | **42.0** | **26.9** | Complementary gains |
| Temporal objectness threshold 0.0 | - | - | ~34 | ~28 | No objectness filtering |
| Temporal objectness threshold 0.3 | - | - | ~42 | ~27 | Large AP50 improvement |

Pre-training data scaling experiment:

| Pre-training Scale | CIDEr (PT) | AP50 (PT+FT) | Recall (PT+FT) |
|-------------|-----------|-------------|----------------|
| 1K | ~20 | ~33 | ~24 |
| 10K | ~30 | ~36 | ~25 |
| 100K | ~40 | ~39 | ~26 |
| 1M | ~50 | **~42** | **~27** |

### Key Findings

- **Pre-training is critical**: Fine-tuning only (FT) achieves AP50 of 15.8, whereas pre-training followed by fine-tuning (PT+FT) reaches 40.0 — a substantial gap.
- **Data scaling remains effective**: From 1K to 1M pre-training videos, all metrics improve continuously with no signs of saturation.
- **Automatic annotation outperforms the baseline but underperforms the trained model**: Using automatic annotation predictions directly (AP50=27.1) is inferior to training GROVE (AP50=33.6), indicating that the model smooths out annotation noise during training.
- **SVO triples outperform full captions** as LLM input — full captions cause the LLM to over-compress its output.
- **Visual trackers in Stage 3 are counterproductive** (CoTracker3 leads to a 3.7% drop in AP50) due to tracking drift caused by viewpoint changes.

## Highlights & Insights

- **Combined paradigm of large-scale data and small-scale precise annotation**: Pre-training on 1M automatically annotated videos followed by fine-tuning on 3.5K manually annotated samples demonstrates a "quantity-first, quality-second" strategy with broad applicability to multi-task vision-language model training.
- **LLM-driven annotation consistency**: Framing the cross-frame noun phrase unification problem as a text classification task cleverly leverages the semantic understanding of LLMs, proving more robust than visual trackers.
- **The temporal objectness head is simple yet effective** — a single additional MLP head addresses the core challenge of object disappearance and reappearance in video grounding.
- The conversion from mask decoder to bounding box decoder achieves efficient knowledge transfer by reusing large-scale pretrained weights.

## Limitations & Future Work

- HowToGround1M is derived from instructional videos (HowTo100M), and its domain bias may limit generalization to other video types.
- GLaMM, used as a frame-level model in the annotation pipeline, may be replaced by stronger video-level captioning models to further improve annotation quality.
- Bounding box localization is less precise than pixel-level segmentation and may be insufficient for elongated or irregularly shaped objects.
- The iGround annotation scale (3.5K videos) remains limited; expanding it may yield further improvements.
- The current approach focuses on human-object interactions in instructional videos; generalization to open-domain video scenarios remains to be validated.

## Related Work & Insights

- GLaMM (image grounded caption generation) is the direct technical foundation of this work; GROVE extends it to the video domain.
- Video temporal grounding methods such as Moment-DETR typically assume a given text query, whereas GROVE simultaneously generates captions and performs grounding.
- The continued utility of HowTo100M / HowToCaption as large-scale video data sources is demonstrated — despite noisy annotations, sufficient data volume can be effectively exploited through training.
- Extracting SVO triples via POS tagging as a text preprocessing step in the automatic annotation pipeline is a technique worth adopting in related work.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The problem formulation and data construction methodology are notably original; the model design is more incremental.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluation on five datasets with detailed data scaling analysis and per-stage ablation of the automatic annotation pipeline.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear structure, well-defined contributions, and rich figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ — The released datasets (HowToGround1M and iGround) and annotation methodology provide lasting contributions to the research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](revisiting_adversarial_patch_defenses_on_object_detectors_unified_evaluation_lar.md)
- [\[CVPR 2026\] Beyond Caption-Based Queries for Video Moment Retrieval](../../CVPR2026/object_detection/beyond_caption-based_queries_for_video_moment_retrieval.md)
- [\[ICCV 2025\] Adversarial Attention Perturbations for Large Object Detection Transformers](adversarial_attention_perturbations_for_large_object_detection_transformers.md)
- [\[ICCV 2025\] LMM-Det: Make Large Multimodal Models Excel in Object Detection](lmm-det_make_large_multimodal_models_excel_in_object_detection.md)
- [\[ICLR 2026\] ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection](../../ICLR2026/object_detection/forestpersons_a_large-scale_dataset_for_under-canopy_missing_person_detection.md)

</div>

<!-- RELATED:END -->
