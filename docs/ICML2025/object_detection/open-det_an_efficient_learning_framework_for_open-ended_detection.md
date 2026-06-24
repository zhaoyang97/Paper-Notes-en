---
title: >-
  [Paper Note] Open-Det: An Efficient Learning Framework for Open-Ended Detection
description: >-
  [ICML 2025][Object Detection][Open-Ended Detection] Open-Det proposes an efficient open-ended detection (OED) framework. By reconstructing the object detector (decoupling one-to-many/one-to-one matching), introducing a VL-prompts distillation module to bridge the vision-language semantic gap, utilizing a LoRa Head + Text Denoising to accelerate LLM training, and applying a Masked Alignment Loss to eliminate contradictory supervision, Open-Det achieves superior detection perfo…
tags:
  - "ICML 2025"
  - "Object Detection"
  - "Open-Ended Detection"
  - "Vision-Language Alignment"
  - "knowledge distillation"
  - "LoRa Head"
  - "Masked Alignment Loss"
date: 2026-05-08
content_hash: d5c1851c824f7cc6
---

# Open-Det: An Efficient Learning Framework for Open-Ended Detection

**Conference**: ICML 2025  
**arXiv**: [2505.20639](https://arxiv.org/abs/2505.20639)  
**Code**: [https://github.com/Med-Process/Open-Det](https://github.com/Med-Process/Open-Det)  
**Area**: Object Detection / Open-Vocabulary Detection / Multimodal VLM  
**Keywords**: Open-Ended Detection, Vision-Language Alignment, knowledge distillation, LoRa Head, Masked Alignment Loss

## TL;DR

Open-Det proposes an efficient open-ended detection (OED) framework. By reconstructing the object detector (decoupling one-to-many/one-to-one matching), introducing a VL-prompts distillation module to bridge the vision-language semantic gap, utilizing a LoRa Head + Text Denoising to accelerate LLM training, and applying a Masked Alignment Loss to eliminate contradictory supervision, Open-Det achieves superior detection performance (APr +1.0%) using only 1.5% of the training data and 20.8% of the training epochs compared to GenerateU.

## Background & Motivation

**Background**: Open-Vocabulary Detection (OVD) has extended detection capabilities from closed-set to open-set, but still relies on an extra category vocabulary as input during inference. Open-Ended Detection (OED), as a more general paradigm, detects and generates object names without predefined categories, with GenerateU being pioneering work in this direction.

**Limitations of Prior Work**: GenerateU suffers from three core issues: (a) demanding large-scale datasets (5.077M) and substantial GPU resources (16 A100 GPUs); (b) slow training convergence (149 epochs); (c) limited detection performance.

**Key Challenge**: The authors dissect three fundamental causes:
   - **Semantic Gap**: Directly feeding visual queries into the LLM results in insufficient alignment between visual and language modalities in the high-dimensional feature space.
   - **Contradictory Supervision**: Neglecting intra-image inter-class relationships leads to contradictory losses and gradients.
   - **Heavy LLM Head + Noisy Alignment**: The 32K vocabulary of T5 corresponds to a head layer with 24.7M parameters. During the initial training phase, the alignment quality of visual queries is poor, and noisy supervision damages the pre-trained weights.

**Goal**: To design an efficient OED framework that accelerates training convergence, improves training efficiency and performance, and removes reliance on large-scale datasets.

**Key Insight**: Accelerate from both the detector and the LLM ends—using decoupled matching in the detector to speed up box training, and using a LoRa Head + VL-prompts distillation in the LLM to accelerate name generation training.

**Core Idea**: Achieve OED performance surpassing GenerateU with minimal data and resources through four collaborative modules (an efficient detector, VL distillation, a denoising LLM, and bidirectional alignment).

## Method

### Overall Architecture

Open-Det comprises four collaborative components: **(1) Object Detector (ODR)**, which generates bounding boxes and visual queries; **(2) Prompts Distiller**, which distills vision-language alignment knowledge from the VLM (CLIP) into VL-prompts via VLD-M; **(3) Object Name Generator**, based on the T5 language model, which uses VL-prompts as inputs to generate object names; and **(4) Vision-Language Aligner**, which enhances bidirectional alignment between visual queries and text embeddings via BVLA-M.

Input pipeline: An image $I \in \mathbb{R}^{H \times W \times 3}$ is fed into the detector to obtain the decoder query $Q_d$, while object names $T$ are processed by the VLM to obtain text embeddings $T_e$. BVLA-M computes alignment scores and matching indices, while VLD-M distills VLM knowledge into VL-prompts $P_{vl}$, which are eventually fed into T5 to generate object names. The inference stage does not require the Vision-Language Aligner, achieving vocabulary-free detection.

### Key Designs

1. **Object Detector (ODR) — Decoupled One-to-Many/One-to-One Matching**:

    - **Function**: Accelerating the training convergence of the box detector.
    - **Mechanism**: The first 4 decoder layers use one-to-many matching + cross-attention to enhance localization ability, while the last 2 layers use one-to-one matching + self-attention to eliminate redundant detections. This achieves decoupling directly inside the decoder without extra branches or heads.
    - **Design Motivation**: Drawing inspiration from DINO's anchor box denoising and one-to-many matching to accelerate convergence, while avoiding the added complexity of extra branches required by existing methods.
    - In addition, a threshold-based query selection method is introduced: $Q_{id} = \{e_t \in E | \sigma(\text{Linear}(e_t)) > \lambda\}$ (default $\lambda = 0.05$), enabling object detection with a flexible number of targets.

2. **Bidirectional Vision-Language Alignment Module (BVLA-M)**:

    - **Function**: Enhancing the alignment quality between visual queries and text embeddings.
    - **Mechanism**: Alignment scores are computed from both V-to-L and L-to-V directions: $S_{align} = \cos(Q_d \times M_{VL}, T_e) + \cos(Q_d, T_e \times M_{LV})$, where $M_{VL}$ and $M_{LV}$ denote the transformation matrices for the two directions, respectively.
    - **Design Motivation**: The channel dimension of visual queries (256) is much smaller than that of text embeddings (768). Directly mapping low dimensions to high dimensions for unidirectional alignment yields suboptimal results. Bidirectional alignment matches the two modalities in both of their respective spaces.

3. **Vision-to-Language Distillation Module (VLD-M)**:

    - **Function**: Distilling the vision-language alignment knowledge of VLM into VL-prompts to bridge the semantic gap between visual queries and image-level text embeddings.
    - **Mechanism**: Interacting query $Q_d$ with backbone features $B$ and encoder features $E$ via deformable cross-attention to adaptively sample background information around targets, transforming region embeddings into image-like representations. VL-prompts $P_{vl}$ are then generated through FFN reweighting, MLP projection, and linear fusion.
    - **Design Motivation**: Distinct from RegionCLIP (which generates pseudo labels) and DVDet (which crops expanded boxes), VLD-M directly enriches background information and distills knowledge at the feature level, which is more efficient.
    - A Cosine Similarity Loss is used to supervise the distillation between $P_{vl}$ and $T_e$.

4. **Object Name Generator — LoRa Head + Text Denoising**:

    - **Function**: Accelerating the training of the LLM (T5) and preventing noisy alignment from degrading the pre-trained weights.
    - **LoRa Head**: Freezing the heavy-weight head of T5 (24.7M parameters) in the initial training stage and introducing a LoRa adapter as a lightweight head to accelerate training, significantly reducing trainable parameters.
    - **Text Denoising**: Adding Gaussian noise $\mathcal{N}(0, \sigma^2)$ (where $\sigma$ is the standard deviation of $T_e$) to text embeddings $T_e$ before feeding them into T5 to reconstruct the text, enhancing model robustness against noisy inputs.
    - **Design Motivation**: Queries are insufficiently trained in the early stage. Poor alignment quality constitutes noisy supervision; directly training all weights of the LLM would corrupt the pre-training.

### Loss & Training

- **Masked Alignment Loss (MAL)**: A binary mask is generated via the text embedding similarity matrix $M = T_e \times T_e^\top$ (with threshold $\tau = 0.99$), labeling same-category targets as 1 to avoid contradictory negative constraints. Unlike ScaleDet, MAL resolves query-text matching conflicts through similarity-binarized BCE updates instead of unifying labels across multiple datasets.
- **Joint Loss**: Jointly optimizes the binary classification score $p_i$, alignment score $s_i$, and IoU score $u_i$:
  $$\mathcal{L}_{JL} = -\frac{1}{N}\sum_{i=1}^{N}[(\sqrt{p_i^\alpha s_i^\alpha u_i^{1-2\alpha}} - p_i)^2 y_i \log(p_i) + p_i^2(1-\sqrt{p_i^\alpha s_i^\alpha u_i^{1-2\alpha}})(1-y_i)\log(1-p_i)]$$
  where $\alpha = 0.25$. The square root operation prevents numerical instability caused by excessively small products of the three scores.

## Key Experimental Results

### Main Results — LVIS MiniVal Zero-shot Transfer

| Method | Backbone | Training Data | Vocab-Free | Epochs | APr | APc | APf | AP |
|------|----------|-----------|------------|--------|-----|-----|-----|-----|
| GLIP(A) | Swin-T | 0.660M | ✗ | - | 14.2 | 13.9 | 23.4 | 18.5 |
| GLIP(C) | Swin-T | 5.456M | ✗ | - | 20.8 | 21.4 | 31.0 | 26.0 |
| Grounding-DINO | Swin-T | 5.460M | ✗ | - | 18.1 | 23.3 | 32.7 | 27.4 |
| GenerateU | Swin-T | 0.077M | ✓ | 149 | 17.4 | 22.4 | 29.6 | 25.4 |
| GenerateU | Swin-T | 5.077M | ✓ | - | 20.0 | 24.9 | 29.8 | 26.8 |
| **Open-Det** | **Swin-T** | **0.077M** | **✓** | **31** | **21.0** | **24.8** | **30.1** | **27.0** |
| **Open-Det** | **Swin-T** | **0.077M** | **✓** | **50** | **21.9** | **25.1** | **30.4** | **27.4** |
| GenerateU | Swin-L | 5.077M | ✓ | - | 22.3 | 25.2 | 31.4 | 27.9 |
| **Open-Det** | **Swin-S** | **0.077M** | **✓** | **31** | **26.0** | **28.6** | **32.8** | **30.4** |
| **Open-Det** | **Swin-L** | **0.077M** | **✓** | **31** | **31.2** | **32.1** | **34.3** | **33.1** |

### COCO & Objects365 Zero-shot Evaluation

| Method | Backbone | Training Data | COCO AP | Objects365 AP |
|------|----------|---------|---------|---------------|
| GenerateU | Swin-L | VG | 33.0 | 10.1 |
| GenerateU | Swin-L | VG+GRIT | 33.6 | 10.5 |
| **Open-Det** | **Swin-L** | **VG** | **35.8 (+2.2)** | **13.8 (+3.3)** |

### Ablation Study — Contribution of Each Component

| ODR | BVLA-M | VLD-M | ONG | Losses | APr | APc | APf | AP |
|-----|--------|-------|-----|--------|-----|-----|-----|-----|
| ✗ | ✗ | ✗ | ✗ | ✗ | 10.2 | 17.4 | 23.2 | 19.6 |
| ✓ | ✗ | ✗ | ✗ | ✗ | 13.9 | 19.8 | 27.6 | 23.1 |
| ✓ | ✓ | ✗ | ✗ | ✗ | 14.7 | 20.3 | 27.9 | 23.5 |
| ✓ | ✓ | ✓ | ✗ | ✗ | 16.3 | 24.2 | 29.9 | 26.3 |
| ✓ | ✓ | ✓ | ✓ | ✗ | 16.9 | 24.5 | 29.7 | 26.3 |
| ✓ | ✓ | ✓ | ✓ | ✓ | **21.0** | **24.8** | **30.1** | **27.0** |

### Key Findings

- **VLD-M contributes the most**: Introducing VLD-M boosts APr from 14.7 to 16.3 (+1.6) and AP from 23.5 to 26.3 (+2.8), proving the critical role of vision-language distillation in bridging the modal gap.
- **Losses yield huge gains for rare classes**: MAL + Joint Loss elevates APr from 16.9 to 21.0 (+4.1), where the Joint Loss alone contributes +2.6 APr, and the Masked Alignment Loss contributes +1.7 APr.
- **LoRa Head + Text Denoising show significant synergy**: Using LoRa Head alone increases APr by +1.2; using Text Denoising alone improves it by +4.6. Combined, they boost APr by +5.7 (from 15.3 to 21.0).
- **Good backbone scalability**: Scaling from Swin-T to Swin-L improves APr substantially from 21.0 to 31.2, demonstrating that the framework can fully exploit the capability of larger backbones.
- **VL Alignment Score**: Open-Det achieves an alignment score of 0.555±0.074 on over 50k object instances, far exceeding GenerateU's 0.448±0.026.

## Highlights & Insights

- **Exceptional Training Efficiency**: Outperforming GenerateU with only 1.5% of the data, 20.8% of the epochs, and 4×V100 GPUs (vs 16×A100). This "small data + modest setup" paradigm is highly valuable for resource-constrained research groups.
- **Decoupled Decoder Design**: Decoupling one-to-many and one-to-one matching inside the decoder layers (first 4 layers vs. last 2 layers) without extra branches. This is simpler than existing methods like Co-DETR, and this trick can be transferred to other DETR-like frameworks.
- **LoRa Head Freezing Strategy**: Freezing the heavy head during early training and replacing it with LoRa—this approach is not limited to OED; any multi-modal training utilizing a "first align, then fine-tune" paradigm can adopt a similar strategy to prevent noisy gradients from corrupting pre-trained weights.
- **Masked Alignment Loss Eliminates Contradictory Supervision**: Using the self-similarity matrix of text embeddings to identify objects of the same category and avoid false negative constraints. This is an elegant and low-cost solution.
- **VLD-M Bridges Region-to-Image Feature Gap**: Adaptively sampling background information via deformable attention to "wrap" regional features into image-level representations. This methodology is transferable to any task involving regional-to-global semantic alignment.

## Limitations & Future Work

- **Cross-Modal Semantic Discrepancy Persists**: The authors candidly acknowledge that the semantic gap between visual regions and text embeddings is not entirely eliminated, restricted by the quality of interactions among the backbone, detector, VLM, and LLM.
- **Trained Solely on VG**: Despite demonstrating data efficiency, performance on more diverse datasets has not been verified, nor did they attempt to incorporate GRIT5M data, which could potentially raise the performance ceiling.
- **Limited LLM Selection**: Utilizing FlanT5-base (relatively small) without exploring larger LLMs like LLaMA, DeepSeek, etc., which might yield substantial improvements.
- **Lack of Segmentation Capability**: Outputting only bounding boxes. The authors suggest integrating a segmentation decoder in the future to construct a unified detection-segmentation framework.
- **Inference Speed Unreported**: Despite emphasizing training efficiency, inference FPS or latency data are not provided, leaving its actual deployment capability unassessed.

## Related Work & Insights

- **vs GenerateU**: Both being OED frameworks, GenerateU directly feeds visual queries to the LLM. Open-Det introduces VL-prompts distillation and bidirectional alignment as an intermediate bridge, achieving superior performance with significantly fewer resources.
- **vs GLIP/Grounding-DINO**: These OVD methods still require category vocabulary input during inference, whereas Open-Det achieves complete vocabulary-free detection and outperforms them using only 1.4% to 11.7% of the training data.
- **vs RegionCLIP/DVDet**: Regarding region-text alignment, RegionCLIP generates pseudo labels and DVDet crops expanded boxes, whereas Open-Det's VLD-M distills directly at the feature level, which is more end-to-end.
- **vs Co-DETR**: In terms of one-to-many matching, Co-DETR employs extra branches, while Open-Det decouples inside the decoder, making it simpler.

## Rating

- Novelty: ⭐⭐⭐⭐ Combined innovation of multiple modules (VLD-M, BVLA-M, MAL, LoRa Head) each has its highlights, but individual technical breakthrough is moderate.
- Experimental Thoroughness: ⭐⭐⭐⭐ Verified on LVIS, COCO, and Objects365 with detailed ablation studies, but lacks comparison on inference speed.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with thorough problem analysis, complete mathematical derivations, and rich figures and tables.
- Value: ⭐⭐⭐⭐ Significant practical value for OED under resource-limited scenarios, though OED itself is still a relatively new direction, and its broader impact remains to be seen.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection](../../AAAI2026/object_detection/vk-det_visual_knowledge_guided_prototype_learning_for_open-vocabulary_aerial_obj.md)
- [\[NeurIPS 2025\] DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection](../../NeurIPS2025/object_detection/dithub_a_modular_framework_for_incremental_openvocabulary_ob.md)
- [\[CVPR 2026\] SRA-Det: Learning Omni-Grained Open-Vocabulary Detection Beyond Category Names](../../CVPR2026/object_detection/sra-det_learning_omni-grained_open-vocabulary_detection_beyond_category_names.md)
- [\[ICLR 2026\] DeCo-DETR: Decoupled Cognition DETR for efficient Open-Vocabulary Object Detection](../../ICLR2026/object_detection/deco-detr_decoupled_cognition_detr_for_efficient_open-vocabulary_object_detectio.md)
- [\[CVPR 2026\] Parameter-Efficient Semantic Augmentation for Enhancing Open-Vocabulary Object Detection](../../CVPR2026/object_detection/parameter-efficient_semantic_augmentation_for_enhancing_open-vocabulary_object_d.md)

</div>

<!-- RELATED:END -->
