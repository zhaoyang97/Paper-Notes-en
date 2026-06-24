---
title: >-
  [Paper Note] Wear-Any-Way: Manipulable Virtual Try-on via Sparse Correspondence Alignment
description: >-
  [ECCV 2024][Human Understanding][Virtual Try-on] The Wear-Any-Way framework is proposed, establishing a strong baseline for high-fidelity virtual try-on based on a dual U-Net diffusion model. By introducing a point control mechanism via Sparse Correspondence Alignment, it enables users to precisely manipulate wearing styles (e.g., rolling up sleeves, opening/closing coats, tucking in hems) through click-and-drag interactions, achieving state-of-the-art performance in both sta…
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Virtual Try-on"
  - "Diffusion Models"
  - "Point Control"
  - "Sparse Correspondence Alignment"
  - "Customizable Generation"
date: 2026-05-08
content_hash: dd91c7c783ffeea9
---

# Wear-Any-Way: Manipulable Virtual Try-on via Sparse Correspondence Alignment

**Conference**: ECCV 2024  
**arXiv**: [2403.12965](https://arxiv.org/abs/2403.12965)  
**Code**: Unreleased (Project Page: mengtingchen.github.io/wear-any-way-page)  
**Area**: Human Understanding  
**Keywords**: Virtual Try-on, Diffusion Models, Point Control, Sparse Correspondence Alignment, Customizable Generation

## TL;DR

The Wear-Any-Way framework is proposed, establishing a strong baseline for high-fidelity virtual try-on based on a dual U-Net diffusion model. By introducing a point control mechanism via Sparse Correspondence Alignment, it enables users to precisely manipulate wearing styles (e.g., rolling up sleeves, opening/closing coats, tucking in hems) through click-and-drag interactions, achieving state-of-the-art performance in both standard and manipulable try-on scenarios.

## Background & Motivation

Virtual Try-on aims to synthesize images of a specific person wearing designated garments, serving as a key technology in fashion e-commerce. Existing methods suffer from two major limitations:

**Insufficient generation quality**: Most methods can only handle simple scenarios (single garments, simple textures), with poor detail preservation for complex textures/patterns. They do not support real-world demands such as model-to-model try-on or multi-garment try-on.

**Uncontrollable wearing styles**: Existing methods lack control over how garments are worn. However, in fashion, alterations in wearing styles (e.g., whether sleeves are rolled up, coats are left open, or tops are tucked in) are crucial for visual presentation.

The core innovation of this work is that, **on top of building a high-quality virtual try-on baseline, it introduces a control-point-based wearing style manipulation mechanism for the first time, allowing users to customize try-on styles with simple click-and-drag interactions**.

## Method

### Overall Architecture

Wear-Any-Way adopts a dual U-Net architecture: the Main U-Net (initialized with Stable Diffusion Inpainting) generates the try-on result, while the Reference U-Net (initialized with standard SD) extracts fine-grained garment features. The overall pipeline consists of two stages: (1) establishing a strong baseline for standard virtual try-on; (2) incorporating point-control capabilities through a Sparse Correspondence Alignment module.

### Key Designs

1. **Strong Baseline: Dual U-Net Virtual Try-on Pipeline**: The foundational architecture for high-fidelity standard try-on.

    - **Main U-Net**: Initialized with the Stable Diffusion inpainting model, taking a 9-channel tensor as input (4-channel latent noise + 4-channel latent of the agnostic person image + 1-channel binary mask).
    - **Reference U-Net**: A standard SD model that takes the garment image as input to extract multi-layer features. Feature injection is achieved by concatenating the reference features' Key/Value in the self-attention layers:
    $\text{Attention} = \text{softmax}\left(\frac{Q_m \cdot \text{cat}(K_m, K_r)^T}{\sqrt{d_k}}\right) \cdot \text{cat}(V_m, V_r)$
    - **CLIP Image Encoder**: Replaces text embeddings to provide global color and texture guidance for the garment.
    - **Pose Control**: Uses DW-Pose to extract person poses, which are encoded by a small CNN and directly added to the Main U-Net's latent noise.
    - **Design Motivation**: The Reference U-Net retains garment fine details (logos, patterns, text, etc.) significantly better than other feature extractors like CLIP/DINOv2/ControlNet, which is validated by experiments.

2. **Sparse Correspondence Alignment**: The core innovation that enables point control.

    - **Point Embedding**: Control points are represented using a disk map $D_{g/p}^{1 \times H \times W}$, where the background is 0 and the point locations are filled with random values from 1 to K (with K=24 as the maximum number of control points). Corresponding points on both the garment image and the person image are assigned the same values. This random assignment decouples semantic meaning from the point indexes, making the point representation permutation-invariant. A convolutional network translates the disk map into high-dimensional embeddings $E_{g/p}^{C \times H \times W}$.
    - **Embedding Injection**: Point embeddings are added to the Query and Key of the attention layers:
    $\text{Attention} = \text{softmax}\left(\frac{(Q_m + E_p) \cdot \text{cat}(K_m + E_p, K_r + E_g)^T}{\sqrt{d_k}}\right) \cdot \text{cat}(V_m, V_r)$
    - **Design Motivation**: By adding point embeddings encoded with identical values to the Q/K of the attention layers, features at the marked locations on the garment image naturally align with the target locations on the person image during generation, thereby achieving precise spatial control.

3. **Training Point Pair Collection**: Resolving the lack of dense garment-to-person correspondence annotations.

    - Utilizing a pre-trained Siamese Stable Diffusion model to extract features from both the person and garment images.
    - Utilizing the feature maps from the last layer and integrating predictions across multiple timesteps to obtain robust matches.
    - Randomly sampling interior and boundary points within the garment region on the person image as queries, and finding corresponding points on the garment image via maximum cosine similarity.
    - Matching from person to garment (since certain points on the flat garment might become invisible after being worn due to complex poses).
    - Comparing SuperGlue, CLIP, DINOv2, Reference U-Net, and SD features, the SD features yield the best matching results.

### Loss & Training

- **Base Loss**: Standard diffusion model MSE noise-prediction loss.
- **Point-weighted Loss**: Increases loss weight around the sampled points to strengthen the supervisory signal for point control.
- **Condition Dropping**: Increases the probability of dropping the pose map and degrades the inpainting mask to a bounding box mask, forcing the model to learn wearing style information from the control points.
- **Zero-initialization**: Adds a zero-initialized convolutional layer at the output of the point embedding network, following the ControlNet approach to achieve progressive integration and improve training stability.

Training configuration:
- 8×A100 GPUs, batch size 64, learning rate 5e-5
- Train the self-attention in the decoder and encoder of the Main U-Net; fully train the Reference U-Net
- Clean training resolution: 768×576 (own dataset) / 512×384 (public dataset comparison)
- 0.3M high-quality try-on triplets (person + top + bottom) data

## Key Experimental Results

### Main Results

**Quantitative Comparison on VITON-HD and DressCode**

| Method | VITON-HD FID↓ | VITON-HD KID↓ | D.C. Upper FID↓ | D.C. Upper KID↓ |
|------|--------------|--------------|-----------------|-----------------|
| VITON-HD | 12.117 | 3.23 | - | - |
| HR-VITON | 11.265 | 2.73 | 13.820 | 2.71 |
| DCI-VTON | 8.754 | 1.10 | 11.920 | 1.89 |
| StableVITON | 8.698 | 0.88 | 11.266 | 0.72 |
| **Wear-Any-Way** | **8.155** | **0.78** | 11.72 | **0.33** |

Achieving best or second-best results on FID and KID (the core metrics for generation quality).

### Ablation Study

**Ablation on Point Embedding Injection and Enhancement Strategies (Landmark Distance↓)**

| Configuration | Dist_upper | Dist_down | Dist_coat |
|------|-----------|-----------|-----------|
| No point control | 35.65 | 21.13 | 43.34 |
| Latent Noise Injection | 27.32 | 16.34 | 30.38 |
| Attention Q,K Injection | 24.35 | 15.79 | 27.27 |
| + Zero-init | 22.65 | 15.33 | 25.56 |
| + Condition-dropping | 18.39 | 12.04 | 20.44 |
| + Point-weighted loss | **17.65** | **10.32** | **20.32** |

**Comparison of Training Point Pair Collection Methods (Landmark Distance↓)**

| Matching Method | Dist_upper | Dist_down | Dist_coat |
|----------|-----------|-----------|-----------|
| SuperGlue | 134.04 | 128.34 | 187.30 |
| CLIP | 93.42 | 89.23 | 129.24 |
| DINOv2 | 83.24 | 70.08 | 103.54 |
| Reference U-Net | 59.34 | 35.23 | 79.98 |
| **Stable Diffusion** | **43.44** | **29.94** | **59.45** |

### Key Findings

- The Reference U-Net is key to maintaining garment details, whereas CLIP, DINOv2, or ControlNet struggle to retain fine patterns like logos and text.
- Attention Q/K injection performs better than Latent Noise injection because it directly establishes correspondences during the feature aggregation stage.
- The three enhancement strategies (zero-init, condition-dropping, point-weighted loss) incrementally contribute to the performance, with condition-dropping bringing the largest improvement.
- Pre-trained SD features significantly outperform traditional matching methods (e.g., SuperGlue) in non-rigid (deformable clothing) matching, validating the semantic correspondence capability of diffusion features.

## Highlights & Insights

1. **Pioneering Interaction Paradigm**: Introduces the concept of click-and-drag control for wearing styles in virtual try-on for the first time, shifting from 'passive generation' to 'active customization'.
2. **Permutation-Invariant Point Embedding Design**: Numerical values are randomly assigned to achieve the permutable property of control points, enabling the model to support an arbitrary number of points at arbitrary locations.
3. **Unified Framework**: A single model can perform multiple tasks—including standard try-on, manipulable try-on, multi-garment try-on, and model-to-model try-on—in a single inference pass.
4. **Clever Use of Diffusion Models for Non-Rigid Matching**: Discovers that pre-trained SD naturally possesses semantic correspondence capabilities, which is leveraged to collect training point pairs.

## Limitations & Future Work

1. The authors note that artifacts may still occur in small regions like hands, which could be mitigated by utilizing higher resolutions or larger models like SDXL.
2. The collection of point pairs relies on the matching quality of pre-trained SD features, which might fail under extreme deformations.
3. Currently only static images are supported; extending this to video try-on is a promising direction.
4. Whether the maximum of K=24 control points is sufficient to cover all fine-grained control requirements remains to be explored further.
5. The metric used to quantitatively evaluate manipulability (landmark distance) relies extensively on the accuracy of the FashionAI detector.

## Related Work & Insights

- **DragGAN/DragDiffusion**: Drag-based image editing methods, which lack precision in clothing scenarios and often distort human body structures.
- **TryOnDiffusion**: Also utilizes a dual U-Net, but requires massive multi-pose training data and lacks support for wearing style control.
- **StableVITON**: Employs zero cross-attention to condition spatial encoders but achieves inferior detail preservation compared to Reference U-Net.
- **Insight**: The semantic correspondence ability of diffusion models can serve as a general-purpose tool for dense matching of non-rigid objects.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ It proposes the concept and complete solution for manipulable virtual try-on for the first time, with an elegantly designed sparse correspondence alignment.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on both standard and controllable dimensions with comprehensive ablations, though lacking user studies.
- **Writing Quality**: ⭐⭐⭐⭐ The illustrations are rich and intuitive, the methodology description is clear, and the application scenarios are well demonstrated.
- **Value**: ⭐⭐⭐⭐⭐ Directly applicable in the fashion industry, opening up a new interaction paradigm for virtual try-on.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VTON 360: High-Fidelity Virtual Try-On from Any Viewing Direction](../../CVPR2025/human_understanding/vton_360_high-fidelity_virtual_try-on_from_any_viewing_direction.md)
- [\[CVPR 2026\] Mobile-VTON: High-Fidelity On-Device Virtual Try-On](../../CVPR2026/human_understanding/mobile_vton_ondevice_virtual_tryon.md)
- [\[CVPR 2026\] RefTon: Reference Person Shot Assist Virtual Try-on](../../CVPR2026/human_understanding/refton_reference_person_shot_assist_virtual_try-on.md)
- [\[CVPR 2025\] Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](../../CVPR2025/human_understanding/reference-free_image_quality_assessment_for_virtual_try-on_via_human_feedback.md)
- [\[ECCV 2024\] GS-Pose: Category-Level Object Pose Estimation via Geometric and Semantic Correspondence](gs-pose_category-level_object_pose_estimation_via_geometric_and_semantic_corresp.md)

</div>

<!-- RELATED:END -->
