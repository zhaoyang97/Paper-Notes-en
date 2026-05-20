---
title: >-
  [Paper Note] OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images
description: >-
  [ICCV 2025][Object Detection][Open-prompt detection] OpenRSD is a general-purpose open-prompt object detection framework for remote sensing that supports both text and image multimodal prompts. It integrates an alignment…
tags:
  - "ICCV 2025"
  - "Object Detection"
  - "Open-prompt detection"
  - "remote sensing object detection"
  - "oriented bounding box detection"
  - "multimodal prompts"
  - "self-training"
date: 2026-05-08
content_hash: b20b0e2e9496d1a5
---

# OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images

**Conference**: ICCV 2025
**arXiv**: [2503.06146](https://arxiv.org/abs/2503.06146)  
**Code**: Unavailable (promised future release)  
**Area**: Object Detection / Remote Sensing
**Keywords**: Open-prompt detection, remote sensing object detection, oriented bounding box detection, multimodal prompts, self-training

## TL;DR

OpenRSD is a general-purpose open-prompt object detection framework for remote sensing that supports both text and image multimodal prompts. It integrates an alignment head and a fusion head to balance speed and accuracy, employs a three-stage training pipeline, and is trained on the ORSD+ dataset comprising 470K images. OpenRSD achieves state-of-the-art average performance across seven public benchmarks while maintaining real-time inference at 20.8 FPS.

## Background & Motivation

Remote sensing object detection faces three major challenges:

**Limitations of closed-set detection**: Most existing work focuses on fixed-category closed-set detection, resulting in poor generalization to novel categories and unseen scenarios.

**Insufficiency of existing open-vocabulary methods**: Prior remote sensing OVD approaches (e.g., DescReg, CastDet, OVA-DETR) are constrained by small-scale datasets and fail to address challenges unique to remote sensing, such as oriented object detection and small object detection.

**Speed–accuracy trade-off**: Deep cross-modal fusion methods (e.g., Grounding DINO) achieve high accuracy but suffer from slow inference, making them impractical for large-scale remote sensing image analysis.

The goal of OpenRSD is to establish a general-purpose remote sensing detection foundation model that supports multimodal prompts, handles both oriented and horizontal bounding box detection, and achieves a favorable balance between accuracy and speed.

## Method

### Overall Architecture

OpenRSD builds upon RTMDet and consists of three core components:
- **Multi-scale image feature encoding**: RTMDet-L backbone + neck for multi-scale feature extraction
- **Prompt construction module**: SkyCLIP (text) and DINOv2 (image) for prompt embedding extraction
- **Multi-task detection heads**: alignment head and fusion head operating in parallel

### Key Designs

1. **Multimodal Prompt Construction**

   Both text and image prompt inputs are supported:
   - **Text prompts**: SkyCLIP text encoder extracts embeddings; GPT-4 generates 10–15 diverse text descriptions per category.
   - **Image prompts**: GT boxes are expanded by a factor of 1.25 and cropped; DINOv2 extracts visual embeddings. A DINOv2+MLP classifier selects the top-100 highest-confidence images as the prompt set.

   Prompt embeddings are projected to a 256-dimensional space via independent MLPs. During training, one prompt type is randomly selected per iteration, with 3–7 embeddings randomly sampled per category. Negative categories are randomly sampled to enhance robustness.

2. **Alignment Head**

   Suited for large-vocabulary fast detection. Classification is performed by computing the similarity between detection features and prompt embeddings:

   $s_i^c = \max_{j \in \{j|l_j=c\}} \left(\alpha \frac{z_i \cdot p_j}{\|p_j\|} + \beta\right)$

   A supervised contrastive loss is additionally introduced to stabilize feature alignment:

   $\mathcal{L}_{\text{ct}} = \sum_i -\frac{1}{|P(i)|} \sum_{j \in P(i)} \log \frac{e^{(z_i^* \cdot p_j / \tau)}}{\sum_{k \in A(i)} e^{(z_i^* \cdot p_j / \tau)}}$

   The total loss is $\mathcal{L}_{\text{aln}} = \mathcal{L}_{\text{ct}} + \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{box}}$, with joint support for horizontal and oriented bounding box regression.

3. **Fusion Head**

   A lightweight cross-modal fusion module is introduced to further enhance detection performance. Three layers of cross-attention enable bidirectional interaction between prompt embeddings and image features:
   - Cross-attention from prompt embeddings to image features
   - Cross-attention from image features back to updated prompt embeddings

   Learnable class embeddings are introduced, mapping randomly assigned category IDs to embeddings, thereby preventing conflicts across tasks of different granularities.

   The total detection loss is $\mathcal{L}_{\text{det}} = \mathcal{L}_{\text{fus}} + \mathcal{L}_{\text{aln}}$.

### Loss & Training

**Three-stage training pipeline**:

1. **Pre-training**: Only the detection module is trained using 7 annotated datasets and Million-AID (region proposals generated by SAM + DINOv2 clustering). Input resolution: 896×896; 360K iterations.
2. **Fine-tuning**: Expanded to 10 datasets of varying granularity with experience-based sampling rates. Input resolution: 832×832; 288K iterations.
3. **Self-training**: The fine-tuned detector generates pseudo-labels covering 200 predefined categories. Labels are filtered via SkyCLIP (similarity threshold 0.24), hard negative mining with category hierarchy trees, and class-agnostic NMS for deduplication. Self-annotated data is mixed with original data at a 1/2 sampling rate for continued training.

## Key Experimental Results

### Main Results

OBB detection results (AP50) on seven public remote sensing datasets:

| Method | DIOR-R | DOTA-v1.0 | DOTA-v2.0 | FAIR1M-2.0 | WHU-Mix | SpaceNet | HRSC2016 | Avg. | FPS |
|---|---|---|---|---|---|---|---|---|---|
| Oriented R-CNN (Swin-T) | 65.8 | 76.0 | 67.4 | 45.0 | 79.3 | 44.6 | 79.1 | 65.3 | 6.9 |
| RTMDet-R-L | 70.1 | 71.5 | 66.7 | 47.6 | 79.9 | 47.4 | 79.9 | 66.2 | 23.8 |
| CastDet* | 63.3 | 72.1 | 63.1 | 37.7 | 77.7 | 45.0 | 74.1 | 61.9 | 6.8 |
| **OpenRSD (Text)** | **73.7** | **76.9** | **70.1** | 46.1 | 79.7 | **50.2** | **88.1** | **69.3** | 20.8 |

In HBB detection, OpenRSD outperforms YOLO-World-L by an average of 8.7% and runs 4× faster than Grounding DINO-T (20.8 vs. 5.4 FPS) at comparable accuracy.

### Ablation Study

**Module ablation**:

| Alignment Head | Fusion Head | Class Embedding | DIOR-R | DOTA-v2.0 | FAIR1M-2.0 | HRSC2016 | Avg. |
|---|---|---|---|---|---|---|---|
| ✓ | | | 71.5 | 68.8 | 45.4 | 84.3 | 63.3 |
| ✓ | ✓ | | 73.1 | 69.5 | 45.3 | 73.1 | 61.8 |
| ✓ | ✓ | ✓ | 72.9 | **70.4** | **45.9** | **84.6** | **64.3** |

**Training stage ablation**: Progressive training from pre-training → fine-tuning → self-training consistently improves average AP from 47.2 → 65.5 → 66.9.

### Key Findings

- **Fusion head yields higher accuracy**: Using the fusion head at inference improves average AP by 1.2% over the alignment head; deep cross-modal interaction enhances generalization under multi-dataset joint training.
- **Text prompts are more stable**: Performance stabilizes with as few as 1 text prompt; image prompts require more samples to achieve progressive improvement.
- **Strong cross-domain generalization**: On 4 unseen datasets, OpenRSD outperforms RTMDet-L by an average of 2.2%, with a 5.2% gain on the small-object benchmark SODA-A.
- **Self-training is effective**: Full-parameter fine-tuning during self-training is optimal, improving HRSC2016 performance from 84.6 to 87.8.

## Highlights & Insights

1. **Dual-head design offers practical flexibility**: The alignment head is fast and vocabulary-scalable; the fusion head achieves higher accuracy—users can choose based on deployment requirements.
2. **The three-stage training strategy** follows a coherent progression of "adaptation → generalization → completion."
3. **Image prompts** reduce reliance on expert knowledge; many fine-grained remote sensing categories (e.g., aircraft subtypes) are difficult to describe accurately with text alone.
4. **The large-scale ORSD+ dataset** comprises 470K images across 200 categories, providing a robust training foundation for open-set remote sensing detection.

## Limitations & Future Work

- Constructing ORSD+ relies on manual integration of multiple specific datasets and pseudo-label generation, limiting scalability to available data sources.
- Extension to denser prediction tasks such as semantic segmentation has not been explored.
- The SkyCLIP filtering threshold (0.24) during self-training is manually set; adaptive threshold selection may yield further improvements.

## Related Work & Insights

- OpenRSD effectively combines the lightweight interaction of YOLO-World with the deep fusion of Grounding DINO, achieving a strong balance in the remote sensing domain.
- The three-stage training strategy may inspire the construction of general-purpose open-set detectors in other domains, such as medical imaging.
- The hybrid prompt paradigm combining image and text inputs is worth broader adoption.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of dual-head design and three-stage training is relatively novel, though individual components are not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Seven evaluation benchmarks, four cross-domain datasets, and detailed ablations—very comprehensive.
- **Writing Quality**: ⭐⭐⭐ Some descriptions contain grammatical errors and redundancy; further polishing is warranted.
- **Value**: ⭐⭐⭐⭐⭐ Real-time remote sensing detection at 20.8 FPS offers high practical deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images](../../CVPR2026/object_detection/small_target_detection_based_on_mask-enhanced_attention_fusion_of_visible_and_in.md)
- [\[CVPR 2026\] Fourier Angle Alignment for Oriented Object Detection in Remote Sensing](../../CVPR2026/object_detection/fourier_angle_alignment_for_oriented_object_detection_in_remote_sensing.md)
- [\[AAAI 2026\] SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection](../../AAAI2026/object_detection/sm3det_a_unified_model_for_multi-modal_remote_sensing_object_detection.md)
- [\[ICCV 2025\] 3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection](3dmood_lifting_2d_to_3d_for_monocular_openset_object_detecti.md)
- [\[ICCV 2025\] Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection](dynamicdino_finegrained_mixture_of_experts_tuning_for_realti.md)

</div>

<!-- RELATED:END -->
