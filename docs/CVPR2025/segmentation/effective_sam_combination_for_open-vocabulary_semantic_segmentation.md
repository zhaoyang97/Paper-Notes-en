---
title: >-
  [Paper Note] Effective SAM Combination for Open-Vocabulary Semantic Segmentation
description: >-
  [CVPR 2025][Segmentation][Open-vocabulary semantic segmentation] This paper proposes ESC-Net, a single-stage open-vocabulary semantic segmentation model. By generating pseudo prompts from CLIP image-text correlation maps and embedding them into pre-trained SAM decoder blocks, the model efficiently leverages SAM's class-agnostic segmentation capability to enhance spatial aggregation. Coupled with a Vision-Language Fusion (VLF) module to achieve precise mask prediction…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Open-vocabulary semantic segmentation"
  - "SAM"
  - "CLIP"
  - "Pseudo prompts"
  - "vision-language fusion"
date: 2026-05-08
content_hash: a048bd166b87fbe2
---

# Effective SAM Combination for Open-Vocabulary Semantic Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2411.14723](https://arxiv.org/abs/2411.14723)  
**Code**: None  
**Area**: Semantic Segmentation / Open-Vocabulary Segmentation  
**Keywords**: Open-vocabulary semantic segmentation, SAM, CLIP, Pseudo prompts, vision-language fusion

## TL;DR
This paper proposes ESC-Net, a single-stage open-vocabulary semantic segmentation model. By generating pseudo prompts from CLIP image-text correlation maps and embedding them into pre-trained SAM decoder blocks, the model efficiently leverages SAM's class-agnostic segmentation capability to enhance spatial aggregation. Coupled with a Vision-Language Fusion (VLF) module to achieve precise mask prediction, ESC-Net achieves SOTA performance on ADE20K, PASCAL-VOC, and PASCAL-Context.

## Background & Motivation

**Background**: Open-vocabulary semantic segmentation aims to perform pixel-level labeling for arbitrary categories. Mainstream methods fall into two categories: (1) two-stage methods that first generate class-agnostic proposal masks using a powerful mask generator (e.g., SAM) and then classify them using CLIP; (2) single-stage methods that directly model image-text correlation to predict segmentation masks (e.g., CAT-Seg).

**Limitations of Prior Work**: Two-stage methods (e.g., OVSeg, SAN) require running the full SAM image encoder (ViT-H), incurring huge computational and memory overhead, alongside domain gap issues when cropped regions are fed into CLIP. Single-stage methods (e.g., CAT-Seg) are more efficient, but CLIP naturally focuses on global semantic alignment rather than local spatial information, leading to low correlation map resolution and imprecise mask boundaries.

**Key Challenge**: SAM possesses powerful spatial aggregation and fine-grained segmentation capabilities, but its two-stage application is highly inefficient. Conversely, single-stage methods are efficient but lack fine-grained spatial segmentation capabilities. A key challenge is whether one can "borrow" SAM's spatial aggregation capability in a single-stage framework without incurring excessive computational overhead.

**Goal**: To design an efficient approach to integrate SAM's segmentation capability into a single-stage open-vocabulary segmentation framework, achieving both accuracy and efficiency.

**Key Insight**: It is observed that SAM's core ability lies in its promptable segmentation framework—even when prompts are ambiguous or point to multiple objects, SAM can still generate valid segmentation masks. Therefore, one can bypass the heavy SAM image encoder and solely utilize SAM's decoder blocks, driven by pseudo prompts generated from CLIP correlation maps.

**Core Idea**: To generate pseudo prompts from CLIP's image-text correlation maps and feed them into pre-trained SAM decoder blocks to enhance the spatial aggregation of CLIP features.

## Method

### Overall Architecture
The inputs to ESC-Net are an image and text descriptions of candidate categories. First, CLIP vision and language encoders extract features $F_v$ and $F_l$, and their cosine similarity is computed to obtain the initial correlation map $C_{v\&l}$. Then, iterative refinement is performed through $N=4$ ESC Blocks: each block first uses a Pseudo Prompt Generator (PPG) to generate pseudo prompts from the correlation map, then uses a pre-trained SAM block to process CLIP image features to enhance spatial aggregation, and finally refines the correlation map using a VLF module. A U-Net style decoder is finally employed to generate the segmentation mask.

### Key Designs

1. **Pseudo Prompt Generator (PPG)**:

    - **Function**: Generates SAM-compatible prompts (point coordinates + masks) for each candidate category from the CLIP image-text correlation maps.
    - **Mechanism**: For each category $n$, a softmax is first applied to the correlation map $C_{v\&l}^n$ to obtain a probability map, which is binarized using a threshold $\alpha$ to obtain approximate object regions. A k-means clustering based on pixel coordinates is then used to divide the regions into $N_o = 5$ object regions (to handle multiple instances of the same category). The probability map is multiplied by the clustered regions to filter out the probability distribution of each object, where the highest probability point acts as the pseudo point prompt and the region mask acts as the pseudo mask prompt. In total, $N_o$ points and $N_o$ masks are generated and encoded via SAM's prompt encoder into sparse and dense prompt features.
    - **Design Motivation**: Traditional SAM relies on precise user prompts, which are unavailable in an open-vocabulary setting. Leveraging CLIP correlation maps as an "approximate localization signal" to generate pseudo prompts is a natural choice. Since SAM is robust to ambiguous prompts, valid segmentations can still be produced even with imprecise prompts.

2. **SAM Block Integration**:

    - **Function**: Leverages the spatial aggregation capability of the pre-trained SAM decoder to enhance CLIP image features.
    - **Mechanism**: The Transformer blocks from the pre-trained SAM mask decoder (including prompt self-attention and bidirectional cross-attention) are extracted to simultaneously take the CLIP image features $F_v$ and the pseudo prompts of each category as inputs. The SAM blocks perform batched processing across each category in parallel, yielding category-wise enhanced image features $(F_v^n)'$, which are subsequently fused back into a unified $F_v'$ via a $1 \times 1$ convolution. Crucially, only SAM's decoder blocks are utilized (omitting the image encoder), resulting in a computational overhead significantly lower than two-stage methods.
    - **Design Motivation**: During pre-training, SAM's decoder has learned how to perform region-level spatial aggregation conditioned on prompts. Directly reusing these pre-trained parameters can inject "how to segment" knowledge into CLIP features without retraining a segmentation network from scratch.

3. **Vision-Language Fusion Module (VLF)**:

    - **Function**: Refines correlation maps utilizing the enhanced image features and text features.
    - **Mechanism**: Formulated in two steps. (a) Image guidance: The correlation map $C_{v\&l}^n$ is embedded via a $1 \times 1$ convolution and concatenated with the enhanced image features $F_v'$, then fed into a Swin Transformer block to generate a visual correlation map $C_v^n$. (b) Text guidance: A linear Transformer (without positional embeddings to remain class-number-agnostic) is used to perform cross-attention between $C_v^n$ and text features $F_l$ to model inter-category relationships, producing the refined correlation map $C_{v\&l}'$. Both steps are computed in parallel across all categories.
    - **Design Motivation**: Since the SAM block only enhances spatial information without directly modeling image-text relationships, VLF supplements it with cross-modal refinement, progressively making the correlation maps more accurate.

### Loss & Training
Standard cross-entropy loss for semantic segmentation is employed. The CLIP encoders and SAM blocks fine-tune their attention layers with a smaller learning rate ($2 \times 10^{-6}$), while other parts use a larger learning rate ($2 \times 10^{-4}$). The model is trained exclusively on COCO-Stuff and evaluated zero-shot on ADE20K, PASCAL-VOC, and PASCAL-Context.

## Key Experimental Results

### Main Results (CLIP ViT-L/14)

| Method | Type | A-847 | PC-459 | A-150 | PC-59 | PAS-20 |
|---|---|---|---|---|---|---|
| CAT-Seg | Single-stage | 16.0 | 23.8 | 37.9 | 63.3 | 97.0 |
| MAFT+ | Single-stage | 15.1 | 21.6 | 36.1 | 59.4 | 96.5 |
| EBSeg | Two-stage (SAM) | 13.7 | 21.0 | 32.8 | 60.2 | 97.2 |
| **ESC-Net** | **Single-stage** | **18.1** | **27.0** | **41.8** | **65.6** | **98.3** |

ESC-Net achieves SOTA performance on all benchmarks, yielding mIoU gains of 2.1 over CAT-Seg on A-847, 3.2 on PC-459, and 3.9 on A-150.

### Ablation Study

| Config | A-847 | PC-459 | A-150 | PC-59 | PAS-20 |
|---|---|---|---|---|---|
| w/o SAM block | 4.8 | 11.7 | 24.2 | 50.4 | 89.4 |
| SAM block (Random Init) | 5.9 | 15.8 | 28.4 | 55.9 | 91.5 |
| SAM block (Pre-trained) | **18.1** | **27.0** | **41.8** | **65.6** | **98.3** |

### Key Findings
- **The pre-trained SAM block is crucial**: A randomly initialized SAM block yields only a marginal improvement (+1.1 on A-847), whereas pre-trained parameters bring a massive gain (+13.3 on A-847), demonstrating the effective transfer of SAM's pre-trained segmentation knowledge.
- **The "point + mask" combination in pseudo prompts is optimal**: Bounding box prompts perform poorly due to the low accuracy and potential overlap of pseudo bounding boxes.
- **Gains are most prominent on datasets with extremely large category numbers (A-847 with 847 classes)**: The improvement (+2.1 mIoU) implies that SAM's spatial aggregation capability assists the model in better distinguishing fine-grained categories.
- **Visualizations reveal** that as the number of ESC Block layers increases, the localization of target objects in correlation maps becomes progressively more precise and dense.

## Highlights & Insights
- **Design trade-offs of "using only the SAM decoder without the encoder"**: This is the most clever aspect of the paper. The SAM image encoder is the computational bottleneck, while the segmentation knowledge primarily resides within the decoder. Bridging CLIP features and the SAM decoder via pseudo prompts yields SAM's segmentation capability with minimal extra overhead.
- **Pseudo prompts as a bridge between two foundation models**: CLIP excels in semantic understanding but remains weak in spatial localization, while SAM excels in spatial segmentation but lacks semantic understanding. Generating pseudo prompts from CLIP correlation maps to drive the SAM block allows both to leverage their respective strengths.
- **Parallel batch implementation**: SAM block computations for $N_c$ categories are parallelized on a batch level, maintaining efficient inference.

## Limitations & Future Work
- The number of k-means clusters in PPG is fixed to $N_o = 5$, which might lack flexibility for scenarios with highly variable instance counts (e.g., crowds vs. a single vehicle).
- The decoder blocks of SAM ViT-B are used; the effects of larger SAM variants (ViT-L/ViT-H) are not evaluated.
- Parallel processing of SAM blocks for all candidate categories implies that memory consumption scales linearly with the number of categories, which may pose memory pressure on extremely large category vocabularies (e.g., A-847).
- Training is only conducted on COCO-Stuff (171 classes); whether training on larger-scale data can bring further improvements remains unverified.

## Related Work & Insights
- **vs CAT-Seg**: Both being single-stage methods, CAT-Seg models CLIP correlations directly; ESC-Net adds pre-trained SAM decoders to enhance spatial information, outperforming it by 2.1 on A-847.
- **vs EBSeg/USE**: These methods also utilize SAM but require the full SAM image encoder (two-stage). ESC-Net outperforms them using only the decoder blocks.
- **vs SAN**: SAN utilizes a side adapter to assist CLIP segmentation, while ESC-Net employs pre-trained SAM blocks as a more robust spatial enhancement component.
- **vs MAFT+**: MAFT+ adapts text embeddings with content-dependent transfer, whereas ESC-Net approaches the task via image spatial enhancement. The two strategies are orthogonal.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of pseudo prompts and the SAM decoder is ingenious, though it remains a novel recombination of existing modules.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six benchmark datasets, two VLM scales, and thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear illustrations and complete methodology descriptions.
- Value: ⭐⭐⭐⭐ Provides a practical reference for efficiently combining foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Semantic Library Adaptation: LoRA Retrieval and Fusion for Open-Vocabulary Semantic Segmentation](semantic_library_adaptation_lora_retrieval_and_fusion_for_open-vocabulary_semant.md)
- [\[CVPR 2025\] Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation](mask-adapter_the_devil_is_in_the_masks_for_open-vocabulary_segmentation.md)
- [\[CVPR 2025\] Exploring Simple Open-Vocabulary Semantic Segmentation](exploring_simple_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] DPSeg: Dual-Prompt Cost Volume Learning for Open-Vocabulary Semantic Segmentation](dpseg_dual-prompt_cost_volume_learning_for_open-vocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](../../ICCV2025/segmentation/training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
