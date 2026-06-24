---
title: >-
  [Paper Note] Improving Medical Multi-modal Contrastive Learning with Expert Annotations
description: >-
  [ECCV 2024][Medical Imaging][Contrastive Learning] Proposes eCLIP, which enhances the representation quality of medical multi-modal contrastive learning without modifying the core CLIP architecture by integrating radiologists' eye-tracking gaze heatmaps as an auxiliary supervisory signal, combined with mixup augmentation and curriculum learning strategies.
tags:
  - "ECCV 2024"
  - "Medical Imaging"
  - "Contrastive Learning"
  - "Expert Annotations"
  - "Eye-tracking Heatmap"
  - "Modality Gap"
date: 2026-05-08
content_hash: 345f94c24678aad2
---

# Improving Medical Multi-modal Contrastive Learning with Expert Annotations

**Conference**: ECCV 2024  
**arXiv**: [2403.10153](https://arxiv.org/abs/2403.10153)  
**Code**: Yes (source code mentioned as available in the paper)  
**Area**: Medical Image  
**Keywords**: Contrastive Learning, Medical Imaging, Expert Annotations, Eye-tracking Heatmap, Modality Gap

## TL;DR

Proposes eCLIP, which enhances the representation quality of medical multi-modal contrastive learning without modifying the core CLIP architecture by integrating radiologists' eye-tracking gaze heatmaps as an auxiliary supervisory signal, combined with mixup augmentation and curriculum learning strategies.

## Background & Motivation

### Core Challenges Faced by Contrastive Learning in the Medical Domain

While vision-language contrastive learning models like CLIP have achieved great success in general domains, they face two core challenges in the medical imaging domain:

**Data Scarcity**: Acquiring medical data involves professional expertise, ethical reviews, and patient privacy, making it difficult to obtain million-scale training data like natural images. Even larger datasets like MIMIC-CXR contain only around 200k chest X-rays.

**Modality Gap**: Embeddings from different modalities (image and text) occupy distinct regions in the shared space, causing the "cone effect". Empirical results show that on the Open-I dataset, the cosine similarities between different anomaly categories (normal, cardiomegaly, atelectasis, opacity) are close to 1, meaning the model struggles to distinguish different medical anomalies.

### Limitations of Prior Work

- **General Pre-trained Models**: CLIP pre-trained on internet data fails to capture fine-grained differences in medical images.
- **Simple Fine-tuning**: Continuing pre-training solely on medical data improves performance but remains limited by finite positive/negative pairs and high intra-modality similarity.
- **Cross-modal Mixup (e.g., $m^2$-mixup)**: Creating hard negative samples by mixing embeddings from different modalities, which may blur the semantic clarity of the embeddings.

### Key Insight

Radiologists' eye-tracking gaze data when reading X-rays contains rich clinical information—eye-tracking heatmaps mark clinical regions of interest (ROIs) that align with descriptions in radiology reports. Leveraging these scarce but high-quality expert annotations to enrich the positive pairs in the training data could significantly improve embedding quality.

## Method

### Overall Architecture

Based on the image and text encoders of standard CLIP, eCLIP introduces an additional **Heatmap Processor**. The overall pipeline keeps the core CLIP architecture unchanged, maintaining plug-and-play generalizability.

For an input sample $(I_i, T_i, E_i)$ (image, text, heatmap), eCLIP generates three embeddings:
- $v_i = f(I_i)$: original image embedding
- $t_i = g(T_i)$: text embedding  
- $v_i^E$: image embedding processed with the expert heatmap

### Key Designs

1. **Heatmap Processor**:

    - **Function**: Fuses the expert eye-tracking heatmap with the original image to generate an augmented image that highlights clinically important regions.
    - **Mechanism**: Converts the image and heatmap into patch sequences and processes them using **Multi-Head Attention (MHA)**—using image patches covered by the heatmap as Queries, and original image patches as Keys and Values. The output is reconstructed back to the original image format and then fed into the standard CLIP image encoder.
    - **Design Motivation**: MHA can adaptively weight different regions based on the attention distribution of the heatmap, which is more flexible than simple mask multiplication or CNN encoding. Ablation studies confirm that MHA outperforms $\odot$ Mask (direct mask multiplication) and CNN encoders.

2. **Mixup Augmentation Strategy**:

    - **Function**: Mixes the original image $I_i$ and the expert-processed image $I_i^E$ to generate $I_i^\lambda = \lambda I_i + (1-\lambda)I_i^E$.
    - **Mechanism**: $\lambda \sim \text{Beta}(\alpha, \alpha)$ with $\alpha=0.3$. The mixed image is passed through the encoder to obtain $v_i^\lambda = f(I_i^\lambda)$, forming a new positive pair $(v_i^\lambda, t_i)$ and corresponding negative pairs.
    - **Design Motivation**: Expert annotation data is extremely scarce (only about 1080 samples). Through mixup, an infinite variety of training samples can be generated from a single expert data instance, effectively expanding the number of high-quality positive pairs. Unlike the cross-modal mixing in $m^2$-mixup, eCLIP performs mixup between the original and expert versions of the same image, preserving semantic consistency.

3. **Curriculum Learning Strategy**:

    - **Function**: Gradually introduces expert annotations in three stages.
    - **Mechanism**:
        - **Cold Start Phase** (first 10% iterations): Expert annotations are not introduced, establishing a robust baseline.
        - **Warm-up Phase** (10% to 40% iterations): The probability of introducing expert samples gradually increases from 0.05 to 0.5.
        - **Cool-down Phase** (40% to 80% iterations): The probability drops to 0.1, balancing foundational training with expert-driven learning.
    - **Design Motivation**: Directly blending scarce expert samples into training leads to instability (the variance of the "naive" baseline in ablation studies is significantly larger). Gradual introduction allows the model to build a stable representation foundation first.

### Loss & Training

**Primary Loss**: An augmented InfoNCE loss that adds extra positive and negative pairs generated by expert embeddings to the standard CLIP loss pairs:

$$\mathcal{L}_{\text{total}} = \frac{1}{2}(\mathcal{L}_{\text{text}} + \mathcal{L}_{\text{image}})$$

**Auxiliary Loss (Priming Loss)**: Trains the heatmap processor to mimic an identity function during the cold start phase:

$$\mathcal{L}_{\text{priming}} = (I_i - I_i^R)^2 \quad \text{当} \quad E_i = \mathbf{1}$$

Total loss: $\mathcal{L} = w_p \cdot \mathcal{L}_{\text{priming}} + (1 - w_p) \cdot \mathcal{L}_{\text{clip}}$, where $w_p = 0.1$.

**Motivation for Priming**: Ensures the heatmap processor degenerates into an identity mapping when no expert annotation is available, so it does not harm the original model's performance and only provides gains when annotations are present.

## Key Experimental Results

### Main Results: Zero-Shot Classification

| Model | CheXpert 5×200 | MIMIC 5×200 | RSNA | CXR 14×100 |
|------|:---:|:---:|:---:|:---:|
| CLIP (Swin Tiny) | 0.517 | 0.452 | 0.808 | 0.169 |
| + naive | 0.532 | 0.452 | 0.807 | 0.167 |
| + DACL | 0.465 | 0.389 | 0.768 | 0.101 |
| + $m^3$-mix | 0.554 | 0.469 | 0.802 | 0.179 |
| + **eCLIP (ours)** | 0.549 | 0.445 | **0.818** | 0.172 |
| + **eCLIP$^P$ (ours)** | **0.558** | **0.463** | **0.819** | **0.192** |
| CLIP (ViT Base) | 0.540 | 0.465 | 0.805 | 0.183 |
| + **eCLIP (ours)** | **0.563** | **0.477** | 0.814 | **0.193** |

eCLIP achieves overall state-of-the-art results on ViT Base, and eCLIP$^P$ (post-trained version) also leads across the board on Swin Tiny.

### Ablation Study

| Configuration | CheXpert F1 | CXR14 F1 | Description |
|------|:---:|:---:|------|
| Base CLIP | 0.517 | 0.169 | Baseline |
| $\odot$ Mask (+E) | 0.540 | 0.165 | Simple mask multiplication |
| CNN Encoder (+E) | 0.534 | 0.163 | CNN-encoded heatmap |
| MHA Encoder (+E) | 0.534 | 0.153 | MHA only, no other techniques |
| MHA (+E,M) | 0.532 | 0.160 | With Mixup |
| MHA (+E,M,C) | 0.545 | 0.173 | With curriculum learning, significant improvement |
| MHA (+rand,M,C,P) | 0.537 | 0.166 | Random heatmap, performance drops |
| MHA (+E,M,C,P) | **0.549** | **0.172** | Full eCLIP |

Curriculum learning (+C) brings the largest gain; the comparison between random and expert heatmaps confirms the critical value of expert annotations.

### Key Findings

- **Cross-modal Retrieval (Open-I)**: eCLIP (ViT Base) achieves R@1/R@5/R@10 = 4.4/10.3/13.5, outperforming CLIP's 3.7/9.2/13.2.
- **RAG Report Generation**: eCLIP achieves BLEU-2 = 0.177 (vs. CLIP 0.172) and a CheXBERT embedding similarity of 0.506 (vs. 0.492), indicating higher-quality retrieved text.
- **Embedding Quality**: eCLIP outperforms the baseline in both uniformity and alignment metrics, narrowing the modality gap.
- **Sample Efficiency**: Under constrained training data scenarios, eCLIP consistently beats CLIP, validating that expert annotations improve data efficiency.

## Highlights & Insights

1. **Plug-and-play Design**: eCLIP does not modify the core CLIP architecture and can be applied to any CLIP variant (Swin Tiny, ViT Small, ViT Base, GLoRIA), showing high practicality.
2. **Efficient Use of Scarce Annotations**: Just 1080 eye-tracking annotated samples significantly improve model quality on ~200k training samples, demonstrating the value of high-quality data over quantity.
3. **Priming Mechanism**: The design rationale that ensures no model degradation in the absence of expert annotations is a valuable reference.
4. **From Retrieval to RAG**: Propagating embedding quality improvements to a frozen LLM's report generation task showcases the practical downstream value of embedding quality.

## Limitations & Future Work

- The volume of expert annotation data is small (1080 samples); the effect of annotation quantity and distribution on performance has not been systematically studied.
- Additional forward passes of expert images during training increase computational overhead.
- Radiology reports generated by RAG have not been clinically validated by medical experts.
- Only expert annotations on the image side are utilized, without extension to the text side (e.g., similar to SimCSE directions).
- Temporal information in eye-tracking data (alignment between fixation sequences and report snippets) is not utilized.

## Related Work & Insights

- **Comparison with GLoRIA**: GLoRIA utilizes local and global features without introducing external expert signals, whereas eCLIP proves the additional value of high-quality external annotations.
- **Difference from Alpha-CLIP**: Alpha-CLIP generates an alpha channel via a segmentation model to guide attention, whereas eCLIP uses real expert gaze data, leading to more accurate semantics.
- **Inspirational Directions**: Extending the paradigm of leveraging expert annotations to other medical tasks requiring specialized domain knowledge (e.g., pathological slides, ultrasound).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of introducing radiologists' eye-tracking data into CLIP learning is novel and meaningful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple tasks such as zero-shot, linear probing, retrieval, and RAG, with comprehensive ablation designs.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, intuitive illustrations, and systematic experimental analysis.
- **Value**: ⭐⭐⭐⭐ Provides a practical solution for utilizing scarce expert annotations in medical multi-modal learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Unsupervised Multi-modal Medical Image Registration via Invertible Translation](unsupervised_multi-modal_medical_image_registration_via_invertible_translation.md)
- [\[CVPR 2026\] TopoCL: Topological Contrastive Learning for Medical Imaging](../../CVPR2026/medical_imaging/topocl_topological_contrastive_learning_for_medical_imaging.md)
- [\[ICCV 2025\] Vector Contrastive Learning for Pixel-wise Pretraining in Medical Vision](../../ICCV2025/medical_imaging/vector_contrastive_learning_for_pixel-wise_pretraining_in_medical_vision.md)
- [\[CVPR 2025\] Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation](../../CVPR2025/medical_imaging/enhanced_contrastive_learning_with_multi-view_longitudinal_data_for_chest_x-ray_.md)
- [\[ECCV 2024\] GTP-4o: Modality-Prompted Heterogeneous Graph Learning for Omni-Modal Biomedical Representation](gtp-4o_modality-prompted_heterogeneous_graph_learning_for_omni-modal_biomedical_.md)

</div>

<!-- RELATED:END -->
