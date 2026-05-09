---
title: >-
  [Paper Note] Advancing Compositional Awareness in CLIP with Efficient Fine-Tuning
description: >-
  [NeurIPS 2025][Multimodal VLM][CLIP compositional reasoning] This paper proposes CLIC, which concatenates two images to form a composite scene and generates hard negatives via cross-image lexical swapping, while constructing multiple positive captions to enhance semantic invariance. By fine-tuning only the CLIP text encoder, CLIC simultaneously improves compositional reasoning (achieving SOTA on SugarCrepe++) and downstream retrieval performance, resolving the long-standing trade-off between compositionality and retrieval in prior methods.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - CLIP compositional reasoning
  - SugarCrepe++
  - image concatenation
  - hard negatives
  - contrastive learning
date: 2026-05-08
content_hash: b5f837e534f4ccbe
---

# Advancing Compositional Awareness in CLIP with Efficient Fine-Tuning

**Conference**: NeurIPS 2025
**arXiv**: [2505.24424](https://arxiv.org/abs/2505.24424)
**Code**: [https://clic-compositional-clip.github.io/](https://clic-compositional-clip.github.io/)
**Area**: Multimodal VLM / CLIP Fine-Tuning
**Keywords**: CLIP compositional reasoning, SugarCrepe++, image concatenation, hard negatives, contrastive learning

## TL;DR

This paper proposes CLIC, which concatenates two images to form a composite scene and generates hard negatives via cross-image lexical swapping, while constructing multiple positive captions to enhance semantic invariance. By fine-tuning only the CLIP text encoder, CLIC simultaneously improves compositional reasoning (achieving SOTA on SugarCrepe++) and downstream retrieval performance, resolving the long-standing trade-off between compositionality and retrieval in prior methods.

## Background & Motivation

**Background**: Vision-language models such as CLIP excel at zero-shot classification and retrieval, yet exhibit significant deficiencies in compositional reasoning. These models tend to learn bag-of-words representations and fail to distinguish "a person in a red shirt riding a grey horse" from "a person in a grey shirt riding a red horse." Several methods (NegCLIP, DAC, TripletCLIP, SVLC) have attempted to improve compositionality by introducing hard negatives.

**Limitations of Prior Work**: The recent SugarCrepe++ benchmark exposes an inconvenient truth: methods previously reported to improve compositionality (e.g., DAC achieving 89.4% on SugarCrepe) perform even worse than the pretrained CLIP on SugarCrepe++ (DAC drops to 53.7%). This indicates that these methods merely learn lexical sensitivity—detecting surface-level token changes—rather than genuinely understanding semantic differences. Furthermore, methods that improve compositionality typically degrade retrieval performance, despite the intuition that stronger compositional understanding should benefit retrieval.

**Key Challenge**: Compositional fine-tuning methods overfit to specific hard negative patterns of particular benchmarks (e.g., fixed swaps of colors or actions), yielding limited gains in genuine semantic understanding. Moreover, aggressive hard negative training tends to corrupt the general-purpose representations learned during CLIP pretraining.

**Goal**: How can CLIP's compositional reasoning be improved without sacrificing—or even while enhancing—its retrieval performance?

**Key Insight**: Rather than generating hard negatives for individual images, CLIC creates a composite scene by concatenating two images and naturally derives hard negatives through cross-image lexical swapping, while leveraging multiple captions per image to construct diverse positive samples that reinforce semantic invariance.

**Core Idea**: Image concatenation + cross-image lexical swapping = low-cost, high-diversity training data for compositional learning.

## Method

### Overall Architecture

At each training iteration, CLIC samples a batch of image-text pairs $\{x_i, y_i\}$, randomly selects a second image of compatible orientation for each sample, concatenates the two into a composite image $u_i$, and constructs 4 positive samples and 1 negative sample from their captions. Only the text encoder is fine-tuned; the visual encoder is frozen. Standard single-image CLIP training is interleaved every other step to prevent drift from the pretrained representation.

### Key Designs

1. **Image Concatenation and Multi-Positive Sample Generation**

   - **Function**: Creates composite scenes and provides rich positive caption descriptions.
   - **Mechanism**: A composite image is formed as $u_i = \text{RandomConcat}(x_i, x_{i+m})$, and four positive samples are constructed: $p_1$ = captions from both images concatenated in order; $p_2$ = captions in reversed order (teaching order invariance); $p_3, p_4$ = randomly selected alternative captions from each image's caption set (increasing descriptive diversity). The number of composites grows quadratically with dataset size, and positive samples naturally cover multiple perspectives of the same image.
   - **Design Motivation**: The compositional space for single-image hard negatives is limited and prone to benchmark-specific overfitting. Image concatenation is itself a compositional operation that naturally produces scenes requiring compositional understanding.

2. **Cross-Image Lexical Swapping for Hard Negative Generation**

   - **Function**: Generates semantically meaningful hard negatives at negligible cost.
   - **Mechanism**: spaCy is used to parse the part-of-speech categories of tokens in $p_1$. A shared lexical category (noun, verb, adjective, etc.) is randomly selected from both sentences, one token from each sentence is chosen, and the two are swapped to produce the negative sample $n$. The resulting caption no longer correctly describes the composite image (unless the swapped tokens happen to be synonymous). Crucially, no specific part-of-speech category is fixed, avoiding overfitting to any particular benchmark.
   - **Design Motivation**: Compared to DAC (which requires LLM generation) and TripletCLIP (which requires synthetic image generation), this approach incurs virtually no additional computational cost and does not target any specific benchmark.

3. **Three-Component Loss Function**

   - **Function**: Separately optimizes contrastive alignment, hard negative discrimination, and semantic invariance.
   - **Mechanism**: The total loss is $\mathcal{L} = \lambda_{Cont}\mathcal{L}_{Cont} + \lambda_{S\text{-}Neg}\mathcal{L}_{S\text{-}Neg} + \lambda_{Uni}\mathcal{L}_{Uni}$. The contrastive loss $\mathcal{L}_{Cont}$ is extended to accommodate 4 positive samples; $\mathcal{L}_{S\text{-}Neg}$ is a binary contrastive loss applied between each positive sample and the hard negative (ensuring the hard negative consistently influences training); $\mathcal{L}_{Uni}$ minimizes the distance between the text embeddings of $p_1$ and $p_2$ (order-swapped captions), teaching the model to produce consistent representations for order-independent but semantically equivalent descriptions.
   - **Design Motivation**: In standard contrastive loss, hard negatives may be overshadowed by easily separable samples within the batch. The dedicated $\mathcal{L}_{S\text{-}Neg}$ term ensures hard negatives always contribute to optimization.

### Loss & Training

Only the text encoder is fine-tuned at $224 \times 224$ resolution. Standard single-image CLIP training is alternated every other step to prevent representation drift. Training uses a subset of approximately 1M samples from PixelProse-RedCaps/CC12M or CogVLM re-captioned LAION, all independent of MS-COCO to ensure fair zero-shot evaluation.

## Key Experimental Results

### Main Results (ViT-B/32, SugarCrepe++ ITT)

| Method | SC++ Replace ITT | SC++ Swap ITT | WG Image | COCO I→T | COCO T→I |
|--------|-----------------|---------------|----------|----------|----------|
| CLIP | 69.5 | 45.7 | 11.0 | 74.1 | 54.6 |
| NegCLIP | 70.5 | 56.4 | 11.0 | 83.6* | 72.2* |
| DAC-LLM | 53.7 | 32.2 | 10.5 | 63.3 | 58.1 |
| TripletCLIP | 73.5 | 43.4 | 11.2 | 72.3 | 56.8 |
| **CLIC-RedCaps** | **76.0** | **61.5** | **12.2** | 76.0 | 59.5 |
| **CLIC-CC12M** | **74.4** | **60.6** | **11.8** | **76.9** | **60.8** |

### Cross-Architecture Generalization (ViT-L/14, CLIPS)

| Method | SC++ Replace ITT | COCO I→T | COCO T→I |
|--------|-----------------|----------|----------|
| CLIPS | 75.5 | 87.3 | 69.9 |
| CLIPS + CLIC | **84.5 (+9%)** | **88.6 (+1.3%)** | **72.1 (+2.2%)** |

### Key Findings

- CLIC is the only method achieving consistent improvements on both SugarCrepe++ and retrieval tasks, resolving the compositionality–retrieval trade-off.
- DAC performs well on SugarCrepe (89.4% Replace ITT) but collapses to 53.7% on SugarCrepe++, revealing that it learns only lexical sensitivity rather than genuine semantic understanding.
- CLIC yields stable improvements across diverse data sources (LAION, CC12M, RedCaps, COCO), demonstrating that the method is not dependent on specific training data.
- Integrating CLIC's visual encoder into LLaVA-1.5-7B improves VQAScore without degrading downstream QA or captioning capabilities.

## Highlights & Insights

- **Image concatenation is an underexplored data augmentation strategy**: Simply concatenating two images creates a new composite scene that inherently demands compositional understanding, with the number of compositions growing quadratically at nearly zero additional cost. This idea is transferable to other tasks requiring compositional reasoning.
- **Cross-image lexical swapping vs. LLM-generated hard negatives**: The former incurs no additional cost and avoids benchmark-specific overfitting, while the latter is expensive and prone to overfitting. CLIC demonstrates that simple methods, within the right framework, can outperform complex ones.
- **SugarCrepe++ exposes "pseudo-compositionality"**: High scores on SugarCrepe are misleading—prior methods merely learned to detect lexical changes rather than understand semantics. This raises important questions about evaluation standards in the field.

## Limitations & Future Work

- Only the text encoder is fine-tuned; joint fine-tuning or adapter-based approaches remain unexplored.
- Cross-image lexical swapping may occasionally exchange semantically equivalent tokens (e.g., synonyms), introducing noise into the negative samples.
- Image concatenation alters aspect ratios and resolution, which may introduce artifacts in high-resolution models.
- Gains on WinoGround are modest (11.0→12.2), indicating room for improvement on scenarios requiring fine-grained spatial reasoning.

## Related Work & Insights

- **vs. NegCLIP**: NegCLIP also employs lexical swapping but operates within a single image caption and targets specific part-of-speech categories (adjectives/nouns). CLIC's cross-image swapping is more natural and is not restricted to any particular category.
- **vs. DAC**: DAC uses LLM/SAM to generate high-quality captions and negatives at substantial computational cost, yet fails on SugarCrepe++, demonstrating that complexity does not imply effectiveness.
- **vs. TripletCLIP**: TripletCLIP additionally generates synthetic images via diffusion models at high cost, and suffers notable degradation in zero-shot classification (ImageNet drops by 8.5%).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The data construction scheme combining image concatenation and cross-image swapping is concise, efficient, and conceptually novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive validation across architectures, data sources, and benchmarks, including LLaVA integration experiments.
- **Writing Quality**: ⭐⭐⭐⭐ The paper is clear and internally consistent, with fair comparative experiments.
- **Value**: ⭐⭐⭐⭐ Resolving the compositionality–retrieval trade-off has important practical implications for CLIP fine-tuning.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] READ: Enhancing Compositional Reasoning in CLIP via Reconstruction and Alignment of Text Descriptions](enhancing_compositional_reasoning_in_clip_via_reconstruction.md)
- [\[ICLR 2026\] Breaking the Limits of Open-Weight CLIP: An Optimization Framework for Self-supervised Fine-tuning of CLIP](../../ICLR2026/multimodal_vlm/breaking_the_limits_of_open-weight_clip_an_optimization_framework_for_self-super.md)
- [\[NeurIPS 2025\] VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning](videorft_incentivizing_video_reasoning_capability_in_mllms_via_reinforced_fine-t.md)
- [\[NeurIPS 2025\] Towards Evaluating Proactive Risk Awareness of Multimodal Language Models](towards_evaluating_proactive_risk_awareness_of_multimodal_language_models.md)
- [\[NeurIPS 2025\] To Think or Not To Think: A Study of Explicit Thinking in Rule-Based Visual Reinforcement Fine-Tuning](think_or_not_think_a_study_of_explicit_thinking_in_rule-based_visual_reinforceme.md)

<!-- RELATED:END -->
