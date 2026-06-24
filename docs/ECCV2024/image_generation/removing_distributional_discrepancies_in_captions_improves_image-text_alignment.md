---
title: >-
  [Paper Note] Removing Distributional Discrepancies in Captions Improves Image-Text Alignment
description: >-
  [ECCV 2024][Image Generation] The authors identify distributional discrepancies (such as word frequency differences) between positive and negative captions at the dataset level, and propose using a text-only classifier to filter out biased data. Fine-tuning LLaVA-1.5 with the debiased dataset yields LLaVA-score, a State-Of-The-Art image-text alignment scoring model.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: 24a198569c8052a7
---

# Removing Distributional Discrepancies in Captions Improves Image-Text Alignment

**Conference**: ECCV 2024  
**arXiv**: [2410.00905](https://arxiv.org/abs/2410.00905)  
**Area**: Image Generation

## TL;DR

The authors identify distributional discrepancies (such as word frequency differences) between positive and negative captions at the dataset level, and propose using a text-only classifier to filter out biased data. Fine-tuning LLaVA-1.5 with the debiased dataset yields LLaVA-score, a State-Of-The-Art image-text alignment scoring model.

## Background & Motivation

- Automatically evaluating the semantic alignment between images and texts is crucial for data cleaning, model evaluation, and generative model improvement.
- Models like CLIP operate in a "bag-of-words" manner, failing to distinguish between "horse eating grass" and "grass eating horse".
- Existing methods construct negative descriptions for contrastive training, but suffer from issues at two levels:
  1. **Instance level**: Prior methods (e.g., NegCLIP) generate disfluent negative descriptions by randomly shuffling word order, which can be easily distinguished by grammar models.
  2. **Distribution level** (first discovered in this work): Even when each negative description is grammatically correct and semantically plausible, there still exist systematic discrepancies in word frequency distribution between positive and negative captions.
- **Typical case**: In the COCO dataset, "giraffe" occurs much more frequently than "elephant". However, when GPT generates negative descriptions, it tends to replace "giraffe" with "elephant", allowing the model to distinguish positive and negative samples using text alone.

## Method

### Overall Architecture

A three-step pipeline: Diverse negative caption generation $\rightarrow$ Distributional discrepancy detection and filtering $\rightarrow$ Vision-language model fine-tuning.

### Key Designs

**1. Hybrid Negative Caption Generation**

Using GPT to generate two types of negative captions:

- **Replacing**: Replacing a linguistic element in the caption with a plausible alternative.
    - Example: "a photo of a broken down stop sign" $\rightarrow$ "a photo of a brand new stop sign"
    - Enhances the model's **perception capability**.

- **Swapping**: Rearranging the linguistic components in the original caption.
    - Example: "an airplane is flying in the blue sky" $\rightarrow$ "a blue airplane is flying in the sky"
    - Enhances the model's **reasoning capability**.

**2. Distributional Discrepancy Detection and Filtering**

Core innovation — exposing discrepancies using a text-only classifier:

1. Train a text-only (ignoring images) BERT binary classifier.
2. Split the dataset into $N$ folds, training and predicting via cross-validation.
3. **Remove the top $k\%$ samples that are correctly predicted with high confidence by the classifier** ($k\%$ for both positive and negative).
4. The remaining data ensures the text-only classifier cannot distinguish positive and negative samples, forcing the image-text alignment model to utilize both visual and textual information.

**Intuition**: If a text-only model can determine whether a caption is positive or negative, it indicates that the sample contains distributional discrepancy features (such as specific word frequency patterns). After removing these biased samples, the trained model truly learns to associate images with texts.

**3. Fine-tuning LLaVA-1.5**

Adopting a simple prompt template: "Does this image I match the following caption T. Answer Yes or No directly."

Alignment score computation:

$$\text{Score} = \frac{e^{P(\text{Yes}|prompt)}}{e^{P(\text{Yes}|prompt)} + e^{P(\text{No}|prompt)}}$$

### Loss & Training

Standard cross-entropy loss with labels as "Yes" (positive samples) or "No" (negative samples). This can also be applied to the ITM head of BLIP2.

Training configuration: batch size=64, 8$\times$A100, 1 epoch, lr=2e-6, filtering ratio $k=30\%$, partition number $N=5$.

## Key Experimental Results

### Main Results

Comprehensive comparison across multiple datasets:

| Method | Winoground-img | Winoground-text | Winoground-group | SeeTRUE-Draw | SeeTRUE-Edit | SugarCrepe-replace | SugarCrepe-swap | MagicBrush |
|------|------|------|------|------|------|------|------|------|
| CLIP-ViT-L-14 | 10.50 | 28.50 | 7.75 | 61.4 | 62.1 | 79.4 | 61.4 | 52.89 |
| NegCLIP | 11.75 | 30.75 | 8.25 | 63.2 | 66.0 | 85.3 | 75.3 | 61.12 |
| BLIP2-ITM | 24.25 | 41.75 | 19.00 | 60.8 | 67.5 | 88.9 | 83.9 | 75.32 |
| Image-Reward | 15.25 | 43.00 | 12.75 | 70.4 | 70.2 | 88.2 | 81.0 | 70.28 |
| VQ2 (PaLI) | 42.25 | 47.00 | 30.50 | 82.6 | 73.6 | — | — | — |
| LLaVA-1.5 (Zero-shot) | 49.75 | 51.00 | 34.25 | 86.9 | 78.3 | 93.5 | 88.3 | 82.61 |
| **LLaVA-score** | **68.00** | **53.75** | **47.25** | **88.8** | 77.7 | **95.3** | **94.9** | **87.28** |

Fine-grained evaluation of attributes, counting, and spatial relations:

| Method | Attribute avg | Counting avg | Spatial Relation avg |
|------|----------|----------|-------------|
| CLIP-ViT-L-14 | 63 | 58 | 53 |
| NegCLIP | 65 | 59 | 57 |
| BLIP2-ITM | 58 | 53 | 51 |
| Image-Reward | 70 | 61 | 57 |
| LLaVA-1.5 | 71 | 62 | 57 |
| **LLaVA-score** | **81** | **71** | **81** |

### Ablation Study

Importance of data filtering strategies:

| Training Settings | Winoground-group↑ | SugarCrepe-replace↑ | SugarCrepe-swap↑ | MagicBrush↑ |
|----------|------------|----------|----------|------------|
| LLaVA-1.5 Baseline | 34.25 | 93.5 | 88.3 | 82.61 |
| Replace-only + Filtering | 38.50 | 95.0 | 89.4 | 84.50 |
| Swap-only + Filtering | 40.25 | 93.2 | 93.0 | 84.75 |
| Hybrid w/o Filtering | 42.00 | 94.8 | 93.8 | 81.50 |
| Random Subsampling (Equivalent data) | 39.75 | 94.1 | 92.5 | 83.00 |
| **Hybrid + Filtering (Full)** | **47.25** | **95.3** | **94.9** | **87.28** |

Impact of filtering ratio $k$: As $k$ increases from 0% to 90%, performance peaks at 30%-40%. The text-only classification accuracy of filtered data drops from 75.9% to approximately 50% (the ideal value), validating that the discrepancy is indeed eliminated.

### Key Findings

1. **Winoground group score surges from 34.25 to 47.25** (+37.9%), on a benchmark widely known for its high difficulty.
2. **Spatial relation understanding improves from 57 to 81** (+42%), which is the most significant gain.
3. Data filtering is critical — without filtering, the performance on MagicBrush is even lower than the baseline (81.50 vs 82.61), indicating that biased data indeed hurts model performance.
4. Replacing and swapping negative captions are complementary — each has its own strengths when used alone, and combining them yields the best results.
5. LLaVA-1.5's zero-shot performance is already the second best (34.25), yet fine-tuning still provides substantial room for improvement.
6. Distributional discrepancy is not unique to GPT — other negative caption generation methods suffer from the same issue.

## Highlights & Insights

- **Core insight is simple yet profound**: Exposing the distributional discrepancy between positive and negative captions using a text-only classifier. This approach can be generalized to any multimodal learning scenario.
- **Method is extremely simple and practical**: It essentially boils down to "training a text classifier + filtering high-confidence samples", requiring no complex architectural design.
- **Discrepancy discovery beyond grammatical level**: Unlike prior works that only check grammatical fluency, this study focuses on implicit statistical properties such as word frequencies.
- **Clear real-world application**: Can be directly applied to the ranking and quality assessment of images generated by text-to-image (T2I) models.

## Limitations & Future Work

- The filtering process requires training BERT classifiers via cross-validation multiple times, language-wise increasing the data preparation overhead.
- Filtering may discard some valuable hard samples, resulting in potential information loss.
- The training data is mainly constructed based on the COCO dataset, and its domain generalization capability remains to be validated.
- The LLaVA-1.5 model is relatively large, and its inference efficiency is lower than lightweight models such as CLIP.
- The optimal filtering ratio $k$ needs to be tuned individually for each dataset.

## Rating

- Novelty: ⭐⭐⭐⭐ — Novel perspective on distributional discrepancies
- Practicality: ⭐⭐⭐⭐⭐ — Simple yet highly effective approach
- Performance: ⭐⭐⭐⭐⭐ — Comprehensively leading across multiple datasets
- Overall Rating: 8.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)
- [\[ECCV 2024\] TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering](textdiffuser-2_unleashing_the_power_of_language_models_for_text_rendering.md)
- [\[ECCV 2024\] Text2Place: Affordance-aware Text Guided Human Placement](text2place_affordance-aware_text_guided_human_placement.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)

</div>

<!-- RELATED:END -->
