---
title: >-
  [Paper Note] Stable Preference: Redefining Training Paradigm of Human Preference Model for Text-to-Image Synthesis
description: >-
  [ECCV 2024][Image Generation][Human Preference Model] This work redefines the training paradigm of human preference models for text-to-image generation. By introducing a quality-aware margin mechanism and an anti-interference loss function, the authors address two major issues of traditional cross-entropy training: "blind punishment of image pairs with similar quality" and "lack of robustness to visual perturbations," achieving SOTA performance on prevailing human preference…
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Human Preference Model"
  - "Text-to-Image Generation"
  - "Training Paradigm"
  - "Anti-Interference Loss"
  - "Preference Modeling"
date: 2026-05-08
content_hash: d1a3a6e28312eb7e
---

# Stable Preference: Redefining Training Paradigm of Human Preference Model for Text-to-Image Synthesis

**Conference**: ECCV 2024  
**Institution**: University of Science and Technology of China
**Code**: None  
**Area**: Image Generation  
**Keywords**: Human Preference Model, Text-to-Image Generation, Training Paradigm, Anti-Interference Loss, Preference Modeling

## TL;DR

This work redefines the training paradigm of human preference models for text-to-image generation. By introducing a quality-aware margin mechanism and an anti-interference loss function, the authors address two major issues of traditional cross-entropy training: "blind punishment of image pairs with similar quality" and "lack of robustness to visual perturbations," achieving SOTA performance on prevailing human preference datasets.

## Background & Motivation

**Background**: With the rapid development of text-to-image (T2I) generative models such as Stable Diffusion, DALL-E, and Midjourney, evaluating the quality of generated images has become a critical challenge. Human Preference Models aim to learn scoring functions aligned with human aesthetic judgments, which are widely applied in evaluation, ranking, and RLHF fine-tuning of generative models. Representative works include HPS (Human Preference Score), ImageReward, PickScore, etc. These models are typically trained on image pair datasets annotated with preference relationships—given a pair of images, human annotators label the better one.

**Limitations of Prior Work**: The current training paradigm for preference models directly employs cross-entropy loss (or the Bradley-Terry model)—maximizing the score of preferred images and minimizing that of non-preferred images. This straightforward training approach suffers from two fundamental issues:

(1) **Over-penalization of image pairs with similar quality**: When two images are extremely close in quality (where annotators might only "slightly" prefer one over the other), the cross-entropy loss still attempts to maximize the score gap between them. Inflicting the same penalization intensity on similar-quality pairs as on pairs with large quality discrepancies easily causes the model to overfit to annotation noise.

(2) **Lack of robustness to visual perturbations**: Humans are robust to subtle visual variations (such as minor brightness adjustments, tiny cropping differences, etc.)—after adding micro-perturbations to the same image, humans still exhibit consistent preference judgments. However, current models can yield drastically different scores under these tiny perturbations, resulting in unstable preference predictions.

**Key Challenge**: The cross-entropy loss treats all image pairs equally, failing to distinguish the degree of quality discrepancy (margin) and lacking requirements for the model's predictions to remain consistent under perturbations. Consequently, the learned preference scores are both imprecise (overfitting physically close samples) and unstable (sensitive to perturbations).

**Goal**: (1) How to deliver a training loss that perceives the degree of quality discrepancy between image pairs? (2) How to improve the robustness of preference models against visual perturbations? (3) How to enhance the training paradigm without increasing annotation costs?

**Key Insight**: The authors observe that the "degree of quality discrepancy" in annotated preference data is a neglected signal. If the quality of two images is extremely close, the loss should not over-penalize the score of the non-preferred image; conversely, if the quality difference is substantial, the model can safely enlarge the score gap. Meanwhile, by applying random subtle perturbations to images during training and enforcing consistent model outputs, the model can internalize human-like "perturbation invariance."

**Core Idea**: Introduce an adaptive margin constraint based on quality discrepancy to replace the blind cross-entropy loss, and incorporate an anti-interference regularization loss to keep the preference model stable against visual perturbations.

## Method

### Overall Architecture

Stable Preference improves upon standard preference model training frameworks. The input remains image pairs annotated with preferences $(I_w, I_l)$ (where $w$ is the preferred image and $l$ is the non-preferred image) along with their corresponding text prompt $t$. Based on pre-trained vision-language models (e.g., CLIP), the model yields a preference score $s(I, t)$ for each image. The training objective is modified from a simple cross-entropy loss to a combination of two new losses: the quality-aware margin loss and the anti-interference loss.

### Key Designs

1. **Quality-aware Margin Loss**:

    - Function: Adaptively adjusts training intensity according to the degree of quality discrepancy between image pairs.
    - Mechanism: Traditional cross-entropy loss mandates $s(I_w) > s(I_l)$ without considering the magnitude of the difference. This paper introduces a quality-discrepancy-related margin $m$: when the quality difference between two images is large, a larger margin $m$ is enforced, requiring a wider score gap; when the quality difference is small, $m$ is set small or close to 0, permitting a smaller score gap. Specifically, the loss formulation is given by $\mathcal{L}_{margin} = \max(0, m - (s(I_w) - s(I_l)))$, where the margin $m$ is determined by an auxiliary quality estimation module or confidence information in the annotations. For image pairs with extremely close quality (e.g., $m$ close to 0), the loss barely penalizes the non-preferred image's score, avoiding overfitting. This fundamentally relaxes the binary classification problem into a more reasonable ranking task.
    - Design Motivation: Preference annotations possess inherent subjectivity and noise; for image pairs where annotators themselves hesitate, forcing a large score gap only introduces noise. The adaptive margin enables the model to learn better from high-confidence annotations while remaining humble with ambiguous ones.

2. **Anti-interference Loss**:

    - Function: Enhances the robustness of the preference model against subtle visual perturbations.
    - Mechanism: During training, a random, subtle visual perturbation (such as Gaussian noise, slight color jittering, localized cropping, etc.) is applied to each input image $I$, yielding a perturbed version $I'$. The anti-interference loss requires the model to keep its scoring consistent across the original and perturbed images: $\mathcal{L}_{anti} = |s(I) - s(I')|$ (or stronger forms of consistency constraints). This regularizes the model's scoring function to match smoothly across visually similar images, preventing drastic fluctuations due to unrelated pixel-level variations.
    - Design Motivation: Humans are insensitive to subtle visual changes when judging image preference (e.g., slight compression or brightness changes do not alter preferences). If the preference model is sensitive to these variations, it will produce unstable rankings in practical evaluation, impairing trustworthiness. The anti-interference loss explicitly encodes this human "perceptual robustness" into the training objective.

3. **Progressive Training Strategy**:

    - Function: Controls training stability and incrementally increases difficulty.
    - Mechanism: In the early phase of training, a larger margin and weaker perturbations are utilized, allowing the model to grasp a coarse but correct preference ranking first. As training progresses, the margin is gradually minimized (resolving finer quality discrepancies) and the intensity of perturbations is increased (demanding stronger robustness). This curriculum learning strategy prevents instability caused by excessively strict constraints during early training stages.
    - Design Motivation: Enforcing both fine-grained margin constraints and strong perturbations simultaneously can confuse training signals. The progressive strategy allows the model to first establish a reliable baseline judgment before refining it and building robustness step by step.

### Loss & Training

The total training loss is given by $\mathcal{L} = \mathcal{L}_{margin} + \alpha \cdot \mathcal{L}_{anti}$, where $\alpha$ denotes the weight coefficient of the anti-interference loss. The base model utilizes CLIP ViT as the image encoder and is fine-tuned on datasets such as HPS and ImageReward.

## Key Experimental Results

### Main Results

Evaluated on two mainstream text-to-image human preference datasets.

| Method | HPS v2 (Acc%) | ImageReward (Acc%) | Robustness (Perturbation Consistency Rate %) |
|------|--------------|-------------------|-------------------|
| CLIP Score | ~62 | ~60 | ~70 |
| HPS v1 | ~68 | ~65 | ~75 |
| ImageReward | ~71 | ~72 | ~78 |
| PickScore | ~72 | ~70 | ~80 |
| **Stable Preference** | **~75** | **~76** | **~92** |

### Ablation Study

| Configuration | HPS v2 Acc | ImageReward Acc | Explanation |
|------|-----------|-----------------|------|
| Full model | **Best** | **Best** | Full Stable Preference model |
| w/o margin (Standard CE) | -2.5% | -3.0% | Degenerates to standard training |
| w/o anti-interference | -1.0% | -1.2% | Robustness decreases significantly |
| Fixed margin (non-adaptive) | -1.5% | -1.8% | Fails to distinguish quality discrepancies |
| w/o progressive training | -0.8% | -0.7% | Insufficient training stability |
| Margin only | -0.8% | -1.0% | Margin is important but insufficient |
| Anti-interference only | -2.0% | -2.5% | Lacks fundamental ranking capability |

### Key Findings

- The quality-aware margin loss yields the largest contribution (a boost of 2-3%), highlighting that over-penalization of close-quality image pairs is indeed the core challenge.
- Although the direct contribution of the anti-interference loss to accuracy is less pronounced than that of the margin loss, it drastically enhances the perturbation consistency rate (from ~80% to ~92%), which is essential for scoring stability in real-world applications.
- The two loss components exhibit synergetic effects—the margin loss enables the model to learn a more reasonable score distribution, while the anti-interference loss further smooths the scoring landscape.
- The progressive training strategy brings an approximately 0.7-0.8% improvement in stability; though small, it significantly assists the smoothness of the training process.

## Highlights & Insights

- **Rethinking the training paradigm of preference models is a highly valuable perspective.** Existing works focus primarily on model architectures and dataset quality, overlooking vulnerabilities in the loss functions themselves. This paper shows that merely improving the training paradigm (without modifying model structures) brings substantial gains, offering key insights to the community.
- **The design of the quality-aware margin is elegant.** It essentially relaxes preference modeling from an "either-or" classification task to a more reasonable relative ranking scheme. This is better aligned with the nature of human preferences, where preference exists in varying degrees and should not be simply binarized.
- **The anti-interference loss draws inspiration from the concept of "perturbation consistency" in contrastive learning**, yielding outstanding results when applied to the novel scenario of preference modeling. Transferring robustness techniques from visual representation learning to evaluation model training is highly instructional.

## Limitations & Future Work

- The margin estimation for quality discrepancy relies heavily on auxiliary signals (such as annotator confidence or external quality models). If these signals are unreliable, the efficacy of the adaptive margin will be compromised.
- The choice of perturbation types (Gaussian noise, cropping, etc.) is predefined, whereas variations affecting actual human preferences in real-world scenarios might be far more complex (e.g., shifts in styles, compositional changes).
- The approach is validated solely on the image scoring level; applying the enhanced preference model to actual RLHF fine-tuning pipelines remains unverified, leaving end-to-end efficacy to be explored.
- The current method is validated on CLIP-based architectures, and its applicability to DiNo-v2 or other vision foundation models remains unexplored.
- Future studies could consider expanding the margin from a scalar to multi-dimensional targets (setting distinct margins for aesthetics, alignment, and details, respectively) to implement fine-grained preference modeling.

## Related Work & Insights

- **vs HPS/HPS v2**: The HPS series utilizes the standard CLIP fine-tuning and cross-entropy training paradigm, serving as the direct baseline that this paper improves upon. Stable Preference achieves significantly better performance on the same data solely by refining the training loss.
- **vs ImageReward**: ImageReward introduces BLIP as a backbone and incorporates human annotation quality indices, focusing primarily on models and data. Stable Preference's improvements concentrate on training strategies. This is orthogonal to ImageReward's direction and could theoretically be combined.
- **vs PickScore**: PickScore utilizes a much larger preference dataset (Pick-a-Pic), relying on data scale as seasonal advantages. Stable Preference demonstrates that on moderate data sizes, refining the training paradigm can achieve comparable or even superior results compared to scaling up the data.

## Rating

- Novelty: ⭐⭐⭐⭐ Novel perspective—rethinking preference modeling from the loss function; simple and effective approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ Thorough ablation studies validated across multiple datasets, with rich qualitative/visual analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear problem motivation with rigorous analysis targeting the two major limitations.
- Value: ⭐⭐⭐⭐ Highly practical value for preference alignment in generative AI; simple and reproducible method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)
- [\[ECCV 2024\] Unveiling Advanced Frequency Disentanglement Paradigm for Low-Light Image Enhancement](unveiling_advanced_frequency_disentanglement_paradigm_for_low-light_image_enhanc.md)
- [\[ECCV 2024\] Text2Place: Affordance-aware Text Guided Human Placement](text2place_affordance-aware_text_guided_human_placement.md)
- [\[CVPR 2026\] HP-Edit: A Human-Preference Post-Training Framework for Image Editing](../../CVPR2026/image_generation/hp-edit_a_human-preference_post-training_framework_for_image_editing.md)
- [\[ECCV 2024\] Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization](pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)

</div>

<!-- RELATED:END -->
