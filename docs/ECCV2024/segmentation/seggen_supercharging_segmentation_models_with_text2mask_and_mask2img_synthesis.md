---
title: >-
  [Paper Note] SegGen: Supercharging Segmentation Models with Text2Mask and Mask2Img Synthesis
description: >-
  [ECCV 2024][Segmentation][data generation] Presents the SegGen data generation framework, reversing the conventional "generate image then label" pipeline to "generate segmentation mask from text first, then generate image from mask," breaking the "chicken-and-egg" bottleneck of segmentation data synthesis, and improving the mIoU of Mask2Former R50 on ADE20K from 47.2 to 49.9 (+2.7).
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "data generation"
  - "Text2Mask"
  - "Mask2Img"
  - "diffusion model"
date: 2026-05-08
content_hash: 74f2e72481e9fe3c
---

# SegGen: Supercharging Segmentation Models with Text2Mask and Mask2Img Synthesis

**Conference**: ECCV 2024  
**arXiv**: [2311.03355](https://arxiv.org/abs/2311.03355)  
**Code**: Not explicitly mentioned  
**Area**: Semantic Segmentation / Data Generation  
**Keywords**: data generation, Text2Mask, Mask2Img, diffusion model, segmentation

## TL;DR
Presents the SegGen data generation framework, reversing the conventional "generate image then label" pipeline to "generate segmentation mask from text first, then generate image from mask," breaking the "chicken-and-egg" bottleneck of segmentation data synthesis, and improving the mIoU of Mask2Former R50 on ADE20K from 47.2 to 49.9 (+2.7).

## Background & Motivation
**Background**: Image segmentation is a pixel-level labeling task with extremely high annotation costs. The scale of mainstream datasets is much smaller than classification datasets (ADE20K has only ~20K training images, COCO has ~118K), where data scarcity limits model performance and generalization capability.

**Limitations of Prior Work**: Prior synthetic data methods (such as DiffuMask, DatasetGAN, and Grounded Diffusion) rely on an "annotator/segmenter module" — essentially another segmentation model — to generate masks for the synthesized images. The performance of downstream segmentation models is thus bounded by the capability ceiling of this annotator, creating a **"chicken-and-egg" dilemma**.

**Key Challenge**: Training a better segmentation model requires better annotation data, but generating better annotation data requires a better segmentation model — a loop that is difficult to break. DiffuMask can only achieve a very low mIoU when trained solely on synthetic data.

**Goal**: Design a data generation method that does not rely on any segmenter/annotator module, fundamentally breaking the chicken-and-egg dilemma.

**Key Insight**: Reverse the pipeline — generate high-quality segmentation masks from text first (Text2Mask), and then generate aligned images conditioned on those masks (Mask2Img). The mask itself functions as the annotation, eliminating the need for an additional annotator.

**Core Idea**: The reversed Text $\to$ Mask $\to$ Image pipeline eliminates the annotator bottleneck. Two complementary data synthesis strategies (MaskSyn for layout diversity and ImgSyn for appearance diversity) jointly boost state-of-the-art segmentation models.

## Method

### Overall Architecture
SegGen consists of two generative models and two data synthesis strategies: (1) extracting textual descriptions from real training images as text prompts using BLIP2; (2) generating new segmentation masks from text prompts using the Text2Mask model; (3) generating aligned images from the masks and text using the Mask2Img model. The two synthesis strategies are: MaskSyn (generating both new masks and new images) and ImgSyn (generating new images based on human-annotated masks). These two complementary strategies allow the synthetic data to be trained alongside real data to improve downstream segmentation models.

### Key Designs
1. **Text2Mask Generative Model**

    - **Function**: Generate highly diverse segmentation masks from text prompts.
    - **Mechanism**: Encode the segmentation mask (pixel values as category IDs) into a three-channel RGB color map (where each color corresponds to a class), directly leveraging the VAE encoding/decoding capabilities of SDXL-base. The model fine-tunes SDXL-base on [text, colormap] pairs. During inference: $\mathbf{C}_{syn} = \text{Text2Mask}(\mathbf{T})$, $\mathbf{M}_{syn} = f_{\text{color} \to \text{mask}}(\mathbf{C}_{syn})$, where $f$ converts the generated colormap back into a mask via nearest-neighbor color matching.
    - **Design Motivation**: Leverage the powerful generative capability of SDXL pre-trained on large-scale data instead of training a mask generator from scratch. The colormap format allows almost lossless reconstruction by the SDXL VAE.

2. **Mask2Img Generative Model**

    - **Function**: Generate highly aligned realistic images conditioned on the segmentation masks and text prompts.
    - **Mechanism**: Adopt the ControlNet architecture by freezing the parameters of SDXL-base and training an additional side network for mask-conditioned image generation. The input is $[\text{text}, \text{colormap}]$: $\mathbf{I}_{syn} = \text{Mask2Img}(\mathbf{T}, \mathbf{C})$. For panoptic/instance segmentation, the boundaries of each segment are outlined with a specific edge color on the colormap to differentiate instances.
    - **Design Motivation**: ControlNet preserves the generalization capability of pre-trained diffusion models while providing precise conditional control. Freezing the main network prevents overfitting to small datasets.

3. **MaskSyn Synthesis Strategy**

    - **Function**: Generate entirely new mask-image training pairs to enhance mask diversity.
    - **Mechanism**: Real image $\to$ BLIP2 extracts caption $\to$ Text2Mask generates new mask $\to$ Mask2Img uses the new mask + caption to generate a new image. Each training sample can be expanded into multiple brand-new (mask, image) pairs.
    - **Design Motivation**: Mask diversity is a major bottleneck in prior works; MaskSyn generates genuinely novel spatial layouts through a generative approach.

4. **ImgSyn Synthesis Strategy**

    - **Function**: Generate diverse new images based on human-annotated masks to enhance image diversity.
    - **Mechanism**: Directly use human-annotated masks and captions as inputs to Mask2Img to generate multiple appearance variations. This acts as an advanced form of data data augmentation.
    - **Design Motivation**: Human-annotated masks offer the highest quality, but their corresponding images have limited diversity. Experiments show that the mask-image alignment of synthetic images can even surpass that of real images (due to inherent imprecisions in human annotations).

### Loss & Training
- **Text2Mask and Mask2Img**: Based on SDXL-base, learning rate $\text{lr}=1\text{e-}5$, 30K iterations, resolution 768.
- **Sampling Settings**: Text2Mask uses a 200-step EDM sampler, Mask2Img uses 40 steps.
- **Synthetic Data Volume**: 10$\times$ expansion for MaskSyn (202K samples) + 50$\times$ expansion for ImgSyn (1.01M samples) on ADE20K; 10$\times$ expansion for ImgSyn only (1.18M samples) on COCO.
- **Training Downstream Segmentation Models**: Two strategies — (1) random data augmentation (replacing with synthetic samples with probability $p_{aug}=60\%$); (2) pre-training on synthetic data followed by fine-tuning on real data.

## Key Experimental Results

### ADE20K Semantic Segmentation (Data Augmentation Strategy)

| Method | Backbone | mIoU (s.s.) | mIoU (m.s.) | Gain |
|------|----------|-------------|-------------|------|
| Mask2Former | R50 | 47.2 | 49.2 | - |
| **+ SegGen** | **R50** | **49.9** | **51.4** | **+2.7/+2.2** |
| Mask2Former | Swin-L | 56.1 | 57.3 | - |
| **+ SegGen** | **Swin-L** | **57.4** | **58.7** | **+1.3/+1.4** |
| Mask DINO | R50 | 48.7 | - | - |
| OneFormer | Swin-L | 57.0 | 57.7 | - |

### COCO Panoptic Segmentation (Synthetic Pre-training Strategy)

| Method | Backbone | PQ | AP_pan^Th | mIoU_pan | Gain |
|------|----------|----|---------|---------|------|
| Mask2Former (real pretrain) | R50 | 52.0 | 42.0 | 61.0 | - |
| **+ SegGen** | **R50** | **52.7** | **43.1** | **62.6** | **+0.7/+1.1/+1.6** |
| Mask DINO (real pretrain) | Swin-L | 58.6 | 50.4 | 67.0 | - |
| **+ SegGen** | **Swin-L** | **59.3** | **51.1** | **68.1** | **+0.7/+0.7/+1.1** |

### Comparison of Training on Pure Synthetic Data

| Method | ADE20K mIoU* | Description |
|------|-------------|------|
| DiffuMask | 18.7 | Previous best synthetic data method |
| **SegGen** | **43.9** | **Massive improvement of +25.2** |

### Ablation Study

| Configuration | ADE20K mIoU (R50) | Description |
|------|-------------------|------|
| Baseline Mask2Former | 47.2 | No synthetic data |
| + MaskSyn only | 48.3 | +1.1 |
| + ImgSyn only | 49.0 | +1.8 |
| + MaskSyn + ImgSyn | **49.9** | +2.7, complementary to each other |

### Key Findings
- SegGen enables Mask2Former R50 to outperform newer model architectures such as Mask DINO and OneFormer.
- In many cases, synthetic images exhibit even better alignment with masks than real images (due to imprecisions in human annotations).
- Models trained on synthetic data demonstrate stronger generalization capabilities to unseen domains (new real-world scenes, AI-generated images).
- ImgSyn contributes more than MaskSyn (+1.8 vs +1.1), but their combination yields the best performance.
- On purely synthetic data, SegGen outperforms DiffuMask by 25.2%, demonstrating the massive value of eliminating the segmenter bottleneck.

## Highlights & Insights
- **"Reversing the pipeline" is the key insight**: Prior methods followed the "generate image $\to$ annotate" pipeline, which was bounded by the capability of the annotator. SegGen's "generate mask $\to$ generate image" idea fundamentally eliminates this bottleneck, representing a paradigm-level innovation. The massive 25.2% improvement on purely synthetic data quantitatively proves this paradigm advantage.
- **Synthetic data can surpass the quality of real annotations**: The alignment between the images generated by ImgSyn and their masks is better than that of real images (due to imperfect human annotations). This is a surprising and inspiring finding — synthetic data can be not only larger in quantity but potentially superior in quality.
- **Strong versatility and practicality**: The same method simultaneously boosts semantic segmentation, panoptic segmentation, and instance segmentation. It is effective across multiple backbones (R50 and Swin-L) without modifying any downstream segmentation model architectures.

## Limitations & Future Work
- The mask colormaps in the Text2Mask model may encounter color confusion when the number of categories is extremely large (e.g., 150 classes in ADE20K, 133 classes in COCO), leading to occasional errors in nearest-neighbor matching.
- The scale of synthetic data is massive (over 1 million samples for ADE20K), which significantly increases training costs.
- The diversity of MaskSyn is bounded by the quality of captions extracted by BLIP2; when descriptions are not detailed enough, the generated masks can drift from the real distribution.

## Related Work & Insights
- **vs DiffuMask**: DiffuMask utilizes the internal features of diffusion models as a segmenter, which is limited by feature quality; SegGen bypasses the annotator completely, exceeding DiffuMask's performance by 25.2% mIoU on purely synthetic data.
- **vs DatasetGAN**: DatasetGAN uses a small segmentation network as an annotator, which is only effective in extreme low-data regimes; SegGen consistently improves state-of-the-art models even under full-training-set settings.
- **vs ControlNet**: SegGen's Mask2Img is based on the ControlNet architecture, but the key innovation lies in introducing the generative capability in the Text2Mask direction.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reversing the data generation pipeline is a paradigm-level innovation, elegantly solving the "chicken-and-egg" dilemma.
- Experimental Thoroughness: ⭐⭐⭐⭐ Well-validated on dual benchmarks (ADE20K & COCO), across three segmentation tasks, multiple backbones, and with training setups ranging from purely synthetic to mixed data.
- Writing Quality: ⭐⭐⭐⭐ The motivation (the chicken-and-egg dilemma) is clearly articulated, and the methodology is comprehensively described.
- Value: ⭐⭐⭐⭐⭐ Establishes a new paradigm for segmentation data synthesis, actually improving the performance of SOTA models with high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Diffusion Models for Open-Vocabulary Segmentation](diffusion_models_for_open-vocabulary_segmentation.md)
- [\[ECCV 2024\] VISA: Reasoning Video Object Segmentation via Large Language Models](visa_reasoning_video_object_segmentation_via_large_language_models.md)
- [\[ICCV 2025\] Learn2Synth: Learning Optimal Data Synthesis Using Hypergradients for Brain Image Segmentation](../../ICCV2025/segmentation/learn2synth_learning_optimal_data_synthesis_using_hypergradients_for_brain_image.md)
- [\[ICCV 2025\] UniGlyph: Unified Segmentation-Conditioned Diffusion for Precise Visual Text Synthesis](../../ICCV2025/segmentation/uniglyph_unified_segmentation-conditioned_diffusion_for_precise_visual_text_synt.md)
- [\[ECCV 2024\] A Semantic Space is Worth 256 Language Descriptions: Make Stronger Segmentation Models with Descriptive Properties](a_semantic_space_is_worth_256_language_descriptions_make_str.md)

</div>

<!-- RELATED:END -->
