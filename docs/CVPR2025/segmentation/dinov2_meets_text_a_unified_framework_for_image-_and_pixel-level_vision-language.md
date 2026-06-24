---
title: >-
  [Paper Note] DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment
description: >-
  [CVPR 2025][Segmentation][DINOv2] This paper proposes dino.txt, which aligns the frozen DINOv2 vision encoder with a text encoder trained from scratch using the LiT strategy. It innovatively uses a concatenation of [CLS] and average-pooled patch tokens as the image representation. Combined with text-image bi-modal data curation, the approach achieves state-of-the-art results on zero-shot classification and open-vocabulary segmentation with only 50K iterations (a fraction of C…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "DINOv2"
  - "Vision-Language Alignment"
  - "LiT"
  - "Zero-Shot Classification"
  - "Open-Vocabulary Segmentation"
date: 2026-05-08
content_hash: 431f6c669737a342
---

# DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment

**Conference**: CVPR 2025  
**arXiv**: [2412.16334](https://arxiv.org/abs/2412.16334)  
**Code**: None (Meta FAIR internal project)  
**Area**: Segmentation / Multimodal VLM  
**Keywords**: DINOv2, Vision-Language Alignment, LiT, Zero-Shot Classification, Open-Vocabulary Segmentation

## TL;DR

This paper proposes dino.txt, which aligns the frozen DINOv2 vision encoder with a text encoder trained from scratch using the LiT strategy. It innovatively uses a concatenation of [CLS] and average-pooled patch tokens as the image representation. Combined with text-image bi-modal data curation, the approach achieves state-of-the-art results on zero-shot classification and open-vocabulary segmentation with only 50K iterations (a fraction of CLIP's training cost).

## Background & Motivation

**Background**: Self-supervised vision foundation models like DINOv2 produce powerful and versatile features, demonstrating excellent performance on downstream tasks such as classification, segmentation, and matching. However, the feature space of these models lacks alignment with language, making them unviable for direct use in tasks requiring a text interface, such as zero-shot recognition and open-vocabulary segmentation. On the other hand, while vision-language models like CLIP align vision and language, they are extremely computationally expensive to train from scratch, and their patch-level dense features are of lower quality than those of self-supervised models.

**Limitations of Prior Work**: Directly applying Locked-image Text Tuning (LiT) to DINOv2 yields subpar results—classification is acceptable, but performance on dense tasks (segmentation, retrieval) is poor. There are two primary reasons: (1) The CLIP/LiT training paradigm only contrasts global image-text representations, preventing gradients from flowing back to patch features; (2) freezing the vision encoder leads to a domain gap between pre-training vision data and LiT training data.

**Key Challenge**: How to add a text interface to DINOv2 at a low cost while preserving its powerful dense feature quality? How to achieve both image-level and pixel-level vision-text alignment under a unified training objective?

**Goal**: (1) A unified framework for both global alignment (classification/retrieval) and dense alignment (segmentation); (2) bridging the domain gap under a frozen vision encoder; (3) an efficient training data strategy.

**Key Insight**: Instead of training from scratch or fine-tuning the DINOv2 vision encoder, one can simply train the text encoder to align using the correct image representation (concatenating CLS and average pooling), supplemented with lightweight learnable vision blocks to bridge the domain gap. Meanwhile, training efficiency is improved through text-image bi-modal data curation.

**Core Idea**: Concatenate the [CLS] token and average-pooled patch tokens as the alignment target, allowing a single loss to simultaneously drive text alignment for both global and dense features.

## Method

### Overall Architecture

The architecture of dino.txt consists of three components: a frozen DINOv2 vision encoder, two layers of learnable vision Transformer blocks (to bridge the domain gap), and a text encoder trained from scratch. An image passes through the frozen DINOv2 to produce a [CLS] token and $N$ patch tokens. After passing through the two layers of learnable vision blocks, the updated [CLS] token and the average-pooled patch tokens are concatenated into a $2D$-dimensional global descriptor $\mathbf{g}$, which is used for contrastive learning against the text [EOS] token. During inference, classification uses the global descriptor $\mathbf{g}$, while segmentation calculates the cosine similarity between each patch's output features and the patch-aligned portion of the text query.

### Key Designs

1. **[CLS]+Average Pooling Concatenation (CLS-AvgPool Concatenation)**:

    - **Function**: Unifies image representation for both global and dense alignment.
    - **Mechanism**: Concatenates the [CLS] token (global semantics) and the average pooling of all patch tokens (aggregation of dense information) into $\mathbf{g} = [\mathbf{c'}; \sigma([\mathbf{f_1'}, \cdots, \mathbf{f_N'}])]$, with a dimension of $2D$. The contrastive loss is computed between $\mathbf{g}$ and the text embedding, allowing gradients to flow back to both the [CLS] and individual patch tokens.
    - **Design Motivation**: Using [CLS] alone yields good classification but poor segmentation, while using average pooling alone yields good segmentation but poor classification. Concatenating both achieves the best of both worlds—this is the most critical finding. Ablation in Table 2 shows that the [CLS avg] combination is optimal across classification, retrieval, and segmentation.

2. **Learnable Vision Blocks**:

    - **Function**: Bridges the domain gap between the pre-training data of the frozen vision encoder and the LiT training data.
    - **Mechanism**: Adds 2 layers of learnable Transformer blocks $\psi$ on top of the frozen DINOv2, preserving the output dimension $D$. These blocks adapt the vision features to the new training data distribution.
    - **Design Motivation**: Direct LiT training on DINOv2 results in poor retrieval and segmentation; adding two vision blocks yields significant improvements with far fewer parameters than fine-tuning the entire vision encoder.

3. **Text and Image Based Curation**:

    - **Function**: Builds a conceptually balanced training dataset.
    - **Mechanism**: Utilizes MetaCLIP's WordNet query balancing strategy on the text side and a hierarchical k-means clustering balancing strategy based on DINOv2 features on the image side (3 levels, with 20M/800K/80K centroids), taking the intersection of both. Approximately 650M pairs are sampled per epoch from a pool of 2.3B image-text pairs.
    - **Design Motivation**: Web-scraped image-text captions are highly noisy. Relying solely on text balancing is insufficient to guarantee uniform coverage of visual concepts. Experiments demonstrate that bi-modal balancing yields a 1-2 % improvement over single-modal balancing.

### Loss & Training

- Employs the standard CLIP contrastive loss with a batch size of 32K-65K.
- Trains for only 50K iterations (seeing about 1.6-3.2B image-text pairs), which is significantly less than CLIP.
- Freezing the vision encoder reduces GPU memory, enabling larger batch sizes.
- Employs a sliding window strategy during segmentation inference; the high-resolution version samples and aggregates crops of different sizes.

## Key Experimental Results

### Main Results

Zero-Shot Classification:

| Method | Vision Encoder | IN1K | IN-v2 | ObjNet |
|------|-----------|------|-------|--------|
| CLIP ViT-L/14 | Trained | 75.3 | 69.8 | 57.1 |
| OpenCLIP ViT-G/14 | Trained | 80.1 | 73.6 | 63.8 |
| **dino.txt ViT-L/14** | **Frozen** | **81.1** | **74.3** | **65.2** |

Open-Vocabulary Segmentation:

| Method | ADE20K | Cityscapes | COCO-Stuff | VOC20 |
|------|--------|-----------|------------|-------|
| TCL | 24.3 | 30.4 | 19.6 | 77.5 |
| GroupViT | 10.6 | 11.1 | 15.3 | 79.7 |
| **dino.txt ViT-L/14** | **29.5** | **40.0** | **24.6** | **73.6** |
| **dino.txt (High-Res)** | **37.2** | - | - | - |

### Ablation Study

| Pooling Method | IN1K Cls | COCO Retrieval | ADE Seg |
|---------|----------|----------|---------|
| [CLS] Only | 78.8 | 30.2 | 8.3 |
| [avg] Only | 74.7 | 32.7 | 13.3 |
| [CLS avg] Concatenation | **79.2** | **34.7** | **18.2** |

| Data Curation Strategy | IN1K | COCO | ADE |
|------------|------|------|-----|
| Text-Only Balancing | 79.2 | 34.7 | 18.2 |
| Image-Only Balancing | 78.9 | 33.9 | 18.0 |
| **Text + Image Bi-balancing** | **80.3** | **37.5** | **20.3** |

### Key Findings

- The [CLS]+average pooling concatenation is the most critical design in this study: classification increases by +4.5 compared to [avg] only, and segmentation increases by +9.9 compared to [CLS] only.
- Adding 2 layers of vision blocks significantly improves segmentation from 18.2 to 20.7, and retrieval from 34.7 to 39.2.
- Bi-modal data curation consistently yields a 1-2 % improvement compared to single-modal curation.
- DINOv2 performs better as a frozen vision encoder for LiT compared to other self-supervised models (MAE, I-JEPA, DINO v1).
- Convergence is achieved in just 50K iterations, showing extreme training efficiency.

## Highlights & Insights

- **Simple and Elegant Unified Representation**: A single concatenation operation simultaneously resolves the conflict between global and dense alignment, eliminating the need for complex multi-task losses or specialized dense alignment modules. The patch features of DINOv2 inherently possess strong spatial localization capabilities; they only require the gradients to flow through.
- **Extreme Training Efficiency**: Requiring only 50K iterations (whereas CLIP requires hundreds of thousands), training is highly efficient thanks to the ultra-large batch sizes enabled by the frozen encoder and the high-quality data curation. This significantly lowers the barrier to entry for vision-language alignment.
- **Importance of Image-Side Data Curation**: The study reveals that the conceptual distributions of text and images in web-scraped pairs are severely mismatched, meaning text-only balancing is insufficient. This insight has strong implications for training any VLM with web-crawled data.

## Limitations & Future Work

- The inference quality for segmentation is highly dependent on the number and resolution of crops in the sliding window strategy; high-resolution inference requires approximately 800 crops and takes about 10 seconds.
- There is still a performance gap in retrieval compared to CLIP, where the frozen encoder may act as a fundamental bottleneck in adapting to new data.
- If DINOv2's patch features are inherently poor at spatial localization for certain concepts, dino.txt cannot resolve this.
- The use of intermediate layer features of DINOv2 for segmentation (multi-scale fusion could yield further gains) was not explored.
- The approach only employs weak supervision (image-text pairs); incorporating semi-supervised training with pixel-level annotations might yield better results.

## Related Work & Insights

- **vs CLIP/OpenCLIP**: CLIP trains both encoders from scratch, which is extremely expensive. dino.txt freezes the vision encoder and only trains the text encoder, increasing efficiency by an order of magnitude while leveraging the powerful dense features from DINOv2.
- **vs CLIPpy**: CLIPpy also attempts to fine-tune an SSL backbone combined with text alignment, but suffers a decline in classification performance. dino.txt avoids this trade-off via [CLS]+avg concatenation.
- **vs MaskCLIP**: MaskCLIP requires special inference adaptations (extracting Value embeddings to bypass attention), whereas dino.txt directly uses final patch tokens for segmentation, offering a simpler design.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The [CLS]+avg concatenation strategy is simple yet surprisingly effective, and the bi-modal data curation provides practical value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluations across three main task types (classification, retrieval, segmentation) with detailed ablated studies.
- **Writing Quality**: ⭐⭐⭐⭐ The arguments are clear, and the motivational chain is robust, though some experimental details are scattered.
- **Value**: ⭐⭐⭐⭐⭐ Unlocks language interfaces for SSL models like DINOv2, offering high practical application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Assessing and Learning Alignment of Unimodal Vision and Language Models (SAIL)](assessing_and_learning_alignment_of_unimodal_vision_and_language_model.md)
- [\[CVPR 2025\] SAIL: Assessing and Learning Alignment of Unimodal Vision and Language Models](assessing_and_learning_alignment_of_unimodal_vision_and_language_models.md)
- [\[NeurIPS 2025\] UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning](../../NeurIPS2025/segmentation/unipixel_unified_object_referring_and_segmentation_for_pixel-level_visual_reason.md)
- [\[CVPR 2025\] Fine-Grained Image-Text Correspondence with Cost Aggregation for Open-Vocabulary Part Segmentation](fine-grained_image-text_correspondence_with_cost_aggregation_for_open-vocabulary.md)
- [\[ACL 2025\] Pixel-Level Reasoning Segmentation via Multi-turn Conversations](../../ACL2025/segmentation/pixel-level_reasoning_segmentation_via_multi-turn_conversations.md)

</div>

<!-- RELATED:END -->
