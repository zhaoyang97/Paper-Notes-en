---
title: >-
  [Paper Note] Learning Representations of Satellite Images From Metadata Supervision
description: >-
  [ECCV 2024][Remote Sensing][Satellite Images] This paper proposes SatMIP (Satellite Metadata-Image Pretraining), which represents satellite image metadata (such as time, geographic location, sensor information, etc.) as text descriptions to align images and metadata in a shared embedding space via an image-metadata contrastive learning task. This constructs satellite image representations that encode both visual features and semantic information. It further introduces SatMIPS…
tags:
  - "ECCV 2024"
  - "Remote Sensing"
  - "Satellite Images"
  - "Metadata Supervision"
  - "Contrastive Learning"
  - "Multimodal Pretraining"
  - "Remote Sensing Representation Learning"
date: 2026-05-08
content_hash: 5875ef6cc0ff2bd7
---

# Learning Representations of Satellite Images From Metadata Supervision

**Conference**: ECCV 2024  
**Paper**: [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3849_ECCV_2024_paper.php)
**Code**: [GitHub](https://github.com/preligens-lab/satmip)  
**Area**: Remote Sensing / Self-Supervised Learning  
**Keywords**: Satellite Images, Metadata Supervision, Contrastive Learning, Multimodal Pretraining, Remote Sensing Representation Learning

## TL;DR

This paper proposes SatMIP (Satellite Metadata-Image Pretraining), which represents satellite image metadata (such as time, geographic location, sensor information, etc.) as text descriptions to align images and metadata in a shared embedding space via an image-metadata contrastive learning task. This constructs satellite image representations that encode both visual features and semantic information. It further introduces SatMIPS (combining image self-supervision and metadata supervision), which outperforms purely visual self-supervised methods like SimCLR on multiple remote sensing downstream tasks.

## Background & Motivation

**Background**: Self-supervised learning (SSL) is increasingly popular in remote sensing to learn general representations from massive unlabeled satellite images. Existing methods are mainly divided into contrastive learning (e.g., SimCLR, MoCo) and masked autoencoding (e.g., MAE), both of which only utilize the images themselves during pretraining. However, remote sensing data naturally comes with rich metadata, such as acquisition time (year, month, day, hour), geographic coordinates (latitude and longitude), sensor type, solar angle, cloud cover, etc.

**Limitations of Prior Work**: Existing remote sensing self-supervised methods ignore the abundant semantic information in metadata. For instance, acquisition time implies seasons and lighting conditions, and geographic coordinates correlate with vegetation types and terrain features. This information is crucial for scene understanding but cannot be utilized by purely visual methods. Some works attempt to use location or time as data augmentation conditions, but they lack a unified framework to fuse heterogeneous metadata.

**Key Challenge**: Remote sensing metadata comes in diverse types (continuous values like coordinates, discrete values like sensor types, timestamps, etc.). How can this heterogeneous information be utilized for pretraining in a unified manner? Directly concatenating multiple types of metadata into a multimodal loss leads to complex and hard-to-scale designs.

**Goal**: (1) How to represent and utilize heterogeneous remote sensing metadata in a unified manner? (2) How to effectively combine metadata supervision with image self-supervision?

**Key Insight**: Inspired by CLIP, the authors propose to represent all metadata uniformly as natural language captions, and then align images and metadata through a CLIP-like image-text contrastive learning framework. This textualization naturally unifies various types of heterogeneous metadata and allows leveraging pre-trained text encoders.

**Core Idea**: Textualize satellite metadata and align it with images through contrastive learning, enabling the representation to encode both visual and semantic information simultaneously.

## Method

### Overall Architecture

The pretraining process of SatMIP resembles CLIP: images are mapped to the embedding space via a visual encoder (such as ResNet-50), while metadata text descriptions are mapped to the same embedding space via a text encoder (such as a pre-trained Sentence Transformer). Then, matched image-metadata pairs are aligned using the InfoNCE contrastive loss. Based on this, SatMIPS incorporates the image-image contrastive loss from SimCLR to simultaneously learn visual invariance and semantic alignment.

### Key Designs

1. **Metadata as Textual Captions**:

    - **Function**: Uniformly converts heterogeneous metadata into text formats to facilitate processing by a text encoder.
    - **Mechanism**: Customizes template-based text descriptions for each metadata type. For example, geographic coordinates (43.6°N, 1.4°E) are converted to "This image was taken at latitude 43.6 degrees north and longitude 1.4 degrees east"; the acquisition time is converted to "This image was captured in July 2021"; the sensor type is converted to "This image was acquired by Sentinel-2". The textual descriptions of various metadata types are concatenated into a complete caption as the metadata representation for the image.
    - **Design Motivation**: Textualization is the most flexible way to handle heterogeneous data—whether they are continuous values, discrete values, or timestamps, they can all be converted into a unified text format. Adding a new metadata type only requires adding a new template, making the framework fully scalable.

2. **Image-Metadata Contrastive Learning (SatMIP Object)**:

    - **Function**: Learns aligned representations of images and metadata in a shared embedding space.
    - **Mechanism**: Employs the InfoNCE contrastive loss, where positive pairs consist of an image and its corresponding metadata description, while negative pairs are the metadata of other images within the batch. The visual encoder learns to extract visual features consistent with the metadata semantics. The loss function is defined as $\mathcal{L}_{SatMIP} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(v_i \cdot t_i / \tau)}{\sum_{j=1}^{N}\exp(v_i \cdot t_j / \tau)}$, where $v_i$ and $t_i$ are the embeddings of the image and metadata respectively, and $\tau$ is the temperature parameter.
    - **Design Motivation**: The contrastive learning framework is mature and effective, directly reusing the CLIP paradigm. Metadata contrast forces the visual encoder to learn features capable of predicting attributes like time and location, which are naturally beneficial for downstream remote sensing tasks.

3. **SatMIPS Combining Image Self-Supervision**:

    - **Function**: Simultaneously learns visual invariance and metadata semantic alignment.
    - **Mechanism**: Incorporates a SimCLR-style image-image contrastive loss on top of SatMIP. Two augmented views are generated for each image and contrastively compared as positive pairs. The total loss is defined as $\mathcal{L}_{SatMIPS} = \mathcal{L}_{SimCLR} + \lambda \mathcal{L}_{SatMIP}$, where $\lambda$ controls the balance between the two objectives. Image self-supervised learning handles local texture and structural features, while metadata supervision handles global semantic features.
    - **Design Motivation**: Pure metadata contrast can only learn semantic features related to the metadata (e.g., distinguishing different geographic regions) but might ignore metadata-unrelated visual details (e.g., building textures). Image self-supervision complements the learning of low-level visual features. Combining both yields a more comprehensive representation.

### Loss & Training

SatMIP utilizes a bidirectional InfoNCE contrastive loss (image-to-text and text-to-image). SatMIPS computes a weighted sum of the SimCLR loss and the SatMIP loss. A pre-trained Sentence-BERT is used as the text encoder, which can be frozen or fine-tuned. ResNet-50 is used as the visual encoder and is trained from scratch. The batch size is 256, and training is performed for 200 epochs.

## Key Experimental Results

### Main Results

| Dataset/Task | Metric | SatMIPS | SimCLR | SeCo | Gain (vs SimCLR) |
|------------|------|---------|--------|------|----------------|
| EuroSAT (Classification) | Top-1 Acc | 96.8 | 95.2 | 94.7 | +1.6 |
| BigEarthNet (Multi-label Classification) | mAP | 88.5 | 86.9 | 87.1 | +1.6 |
| UC Merced (Classification) | Top-1 Acc | 94.3 | 92.8 | 92.1 | +1.5 |
| fMoW (Functional Classification) | Top-1 Acc | 62.7 | 60.1 | 59.4 | +2.6 |

### Ablation Study

| Configuration | EuroSAT Acc | BigEarthNet mAP | Description |
|------|------------|----------------|------|
| SimCLR (baseline) | 95.2 | 86.9 | Pure image self-supervision |
| SatMIP only | 95.9 | 87.8 | Pure metadata contrast |
| SatMIPS | 96.8 | 88.5 | Image + Metadata combination |
| SatMIPS w/o temporal metadata | 96.1 | 87.9 | Temporal information contributes significantly |
| SatMIPS w/o spatial metadata | 96.3 | 88.0 | Spatial/position metadata is also helpful |
| Multimodal classification (Image + Metadata) | 97.2 | 89.1 | Metadata is also used during inference |

### Key Findings

- SatMIPS consistently outperforms purely visual self-supervised methods (SimCLR), proving that metadata supervision is a valuable complementary signal.
- SatMIP alone (pure metadata contrast) already learns non-trivial representations, indicating that metadata indeed encodes crucial scene semantics.
- Multimodal inference (utilizing metadata during inference) further boosts performance, and the SatMIP framework natively supports this mode.
- Compared to SimCLR, SatMIPS converges faster, as metadata provides additional gradient signals to accelerate learning.
- Both temporal and spatial metadata make independent contributions, with their combination yielding the best performance.

## Highlights & Insights

- **Metadata textualization** is a clever and practical design—simplifying the complex problem of "how to uniformly process heterogeneous metadata" into "how to write templates". This strategy can be directly transferred to any domain carrying metadata (such as patient information in medical imaging, weather conditions in autonomous driving, etc.).
- **Redefining the source of supervision signals in remote sensing SSL** is the most significant insight of this work—metadata is free, abundant, and semantically rich, representing a zero-cost "weak annotation" for SSL. This perspective is worth expanding to other data-intensive fields.
- The framework is highly extensible; adding new metadata types only requires adding text templates, without modifying the model architecture.

## Limitations & Future Work

- The design of text templates is relatively simple and fixed, leaving the exploration of using LLMs to automatically generate richer metadata descriptions for future work.
- Metadata quality and completeness affect performance—missing or noisy metadata will discount the efficacy.
- The method is only validated on ResNet-50, and more modern architectures like ViT have not been explored.
- Downstream evaluations are dominated by classification tasks, lacking verification on more complex tasks such as object detection and semantic segmentation.
- Hierarchical combinations of metadata and natural language descriptions (e.g., locating the region via metadata first, then describing the scene content via text) have not been explored.

## Related Work & Insights

- **vs SeCo (Temporal Contrastive Learning)**: SeCo contrastively compares image pairs taken at different times at the same location but does not explicitly use temporal information as supervision. SatMIP directly compares time as text labels, exploiting temporal semantics more thoroughly.
- **vs GeoCLIP**: GeoCLIP focuses on aligning spatial location information with images, whereas SatMIP uniformly processes all metadata types, making it more general.
- **vs CLIP**: SatMIP transfers the image-text contrastive paradigm of CLIP to the remote sensing metadata domain, validating the effectiveness of this paradigm in new areas.

## Rating

- **Novelty**: ⭐⭐⭐⭐ While combining metadata textualization with contrastive learning is intuitively simple, it represents an important paradigm shift in remote sensing SSL.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Verified across multiple remote sensing benchmarks, with ablation studies analyzing the contributions of different metadata.
- **Writing Quality**: ⭐⭐⭐⭐ Clearly presented methodology and distinct contributions.
- **Value**: ⭐⭐⭐⭐ Opens up a new direction for metadata utilization in remote sensing SSL, with high framework generality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] WildSAT: Learning Satellite Image Representations from Wildlife Observations](../../ICCV2025/remote_sensing/wildsat_learning_satellite_image_representations_from_wildlife_observations.md)
- [\[ECCV 2024\] Weakly-Supervised Camera Localization by Ground-to-Satellite Image Registration](weakly-supervised_camera_localization_by_ground-to-satellite_image_registration.md)
- [\[ECCV 2024\] Masked Angle-Aware Autoencoder for Remote Sensing Images](masked_angle-aware_autoencoder_for_remote_sensing_images.md)
- [\[CVPR 2026\] GeoSANE: Learning Geospatial Representations from Models, Not Data](../../CVPR2026/remote_sensing/geosane_learning_geospatial_representations_from_models_not_data.md)
- [\[CVPR 2026\] Spectrally Distilled Representations Aligned with Instruction-Augmented LLMs for Satellite Imagery](../../CVPR2026/remote_sensing/spectrally_distilled_representations_aligned_with_instruction-augmented_llms_for.md)

</div>

<!-- RELATED:END -->
