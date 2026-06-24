---
title: >-
  [Paper Note] Learning Visual Composition through Improved Semantic Guidance
description: >-
  [CVPR 2025][Interpretability][Visual Compositionality] This paper proposes to significantly enhance the visual compositional understanding of standard CLIP models by improving the semantic supervision signals of training data (regenerating high-quality captions using foundation models and replacing training-from-scratch with a pre-trained text encoder). This improves performance on the ARO benchmark from CLIP's 59%/63% to 92%/94%, and on DOCCI image retrieval from 58.4% to 94…
tags:
  - "CVPR 2025"
  - "Interpretability"
  - "Visual Compositionality"
  - "CLIP"
  - "Contrastive Learning"
  - "Recaptioning"
  - "Semantic Guidance"
date: 2026-05-08
content_hash: 34e4f7cac58d1eab
---

# Learning Visual Composition through Improved Semantic Guidance

**Conference**: CVPR 2025  
**arXiv**: [2412.15396](https://arxiv.org/abs/2412.15396)  
**Code**: None  
**Area**: Interpretability  
**Keywords**: Visual Compositionality, CLIP, Contrastive Learning, Recaptioning, Semantic Guidance

## TL;DR

This paper proposes to significantly enhance the visual compositional understanding of standard CLIP models by improving the semantic supervision signals of training data (regenerating high-quality captions using foundation models and replacing training-from-scratch with a pre-trained text encoder). This improves performance on the ARO benchmark from CLIP's 59%/63% to 92%/94%, and on DOCCI image retrieval from 58.4% to 94.5% recall@1, without requiring any architectural modifications.

## Background & Motivation

**Background**: Multimodal contrastive learning models like CLIP have achieved immense success, but a widely recognized drawback is that they essentially treat images as a "bag of words"—failing to understand the composition of attributes and relationships between objects. For instance, "a horse eating grass" and "grass eating a horse" are almost indistinguishable in the embedding space of CLIP.

**Limitations of Prior Work**: To address this compositionality issue, prior works have either designed complex custom architectures (such as the cross-modal encoders in X-VLM or two-stage classification in BLIP) or introduced localization information via multi-task learning. Although effective, these methods are architecturally complex, scale poorly, and rely on high-quality annotated data that is difficult to acquire at scale.

**Key Challenge**: Are the model architecture (ViT) and training objective (contrastive learning) already powerful enough? If so, the fundamental cause of compositionality failure may lie not in the architecture but in the data—specifically, the poor quality of training text labels.

**Goal**: Verify the hypothesis that "improving semantic supervision signals is sufficient for standard CLIP to capture visual compositionality," and find a simple, scalable implementation.

**Key Insight**: The authors hypothesize that ViT has sufficient parameters and scale to capture visual compositionality, and contrastive learning provides an adequate training signal. The key bottleneck lies in the lack of rich target semantic embeddings. Web alt-texts average only 7 words and are filled with noise, which represents the fundamental obstacle to understanding compositionality.

**Core Idea**: Without modifying the architecture or loss function, merely using foundation models to recaption training data and incorporating a strong pre-trained text encoder allows standard CLIP to acquire powerful visual compositional understanding.

## Method

### Overall Architecture

The method is based on the standard dual-tower CLIP architecture, where the image encoder is a ViT-Base (86M parameters), trained using a single contrastive learning loss. Only two key modifications are made: (1) Gemini 1.5 Flash is utilized to regenerate high-quality descriptions for 1B training images; (2) Pre-trained Gemini 1.5 Flash-8B or Gemma2-2B is used as the text encoder (with the last 4 layers unfrozen) to replace training-from-scratch. Training is performed for 150K steps with a global batch size of 65,536, followed by 500 steps of fine-tuning using hard negative data.

### Key Designs

1. **Grounded Recaptioning based on Foundation Models**:

    - **Function**: Replaces web alt-texts with high-quality, detailed image descriptions.
    - **Mechanism**: The original image, alt-text, and web page title are provided to Gemini 1.5 Flash, prompting the model to generate a new description. The generated descriptions average 57 words, which is 8 times longer than alt-texts. Key designs include: (a) raw alt-text and page title provide grounding information to reduce hallucination; (b) the model can perform OCR on text within the image to correct erroneous alt-texts. The median log-likelihood of descriptions significantly increases from -223 for alt-texts to -83, indicating they are much closer to natural language.
    - **Design Motivation**: Noise in alt-texts (e.g., "bigtimerush nyc 007") is the direct cause of contrastive learning failures. Recaptioned descriptions contain rich compositional information, such as object relationships and attribute binding, providing a sufficiently rich supervision signal for contrastive learning.

2. **Pre-trained Strong Text Encoder Replacement**:

    - **Function**: Provides better text representation capabilities to encode rich descriptions.
    - **Mechanism**: Replaces the text encoder trained from scratch with a pre-trained Gemini 1.5 Flash-8B (or Gemma2-2B). Most layers are frozen, and only the last 4 layers are unfrozen to balance computational cost and performance. The original model uses unidirectional attention, while the unfrozen layers are switched to bidirectional attention. Total trainable parameters are 653M.
    - **Design Motivation**: Text encoders trained from scratch struggle to fully comprehend the compositional semantics in long and complex recaptioned descriptions. Pre-trained models already possess deep language understanding capabilities and can better encode compositional information, such as attribute-object relationships.

3. **Data Augmentation Strategies**:

    - **Function**: Further enhances compositional understanding and retrieval robustness.
    - **Mechanism**: Two augmentation methods: (a) **Sentence Sampling**—randomly selecting a subset of sub-sentences from the description as the training target, encouraging the model to focus on local semantics; (b) **Hard Negative Synthesis**—using a foundation model to generate 2 million (later expanded to 64 million) "hard negatives," mimicking the style of relation/attribute swaps in the ARO benchmark. These negative samples are used during the fine-tuning phase.
    - **Design Motivation**: Although long descriptions are informative, they can drown out local compositional information. Sentence sampling forces the model to learn to focus on arbitrary details. Hard negatives directly train the model to distinguish pairs of descriptions that are semantically subtle but have completely different meanings.

## Key Experimental Results

### Main Results

| Method | ARO Relations | ARO Attributes | SugarCrepe Avg | DOCCI Recall@1 |
|------|--------------|----------------|----------------|----------------|
| CLIP | 59% | 63% | ~72% | 58.4% |
| NegCLIP | 71% | 81% | - | - |
| X-VLM | 73% | 87% | - | - |
| MATE | - | - | - | 73.4% |
| **Ours** | **92%** | **94%** | **~93%** | **94.5%** |

### Ablation Study

| Configuration | COCO R@1 | DOCCI R@1 | Description |
|------|----------|-----------|------|
| Train from scratch + alt-text | 47.8 | 53.5 | Baseline |
| Train from scratch + Recaptioned | 46.5 | 75.6 | Recaptioned only: DOCCI +41% |
| Pre-trained encoder + alt-text | 48.3 | 67.2 | Encoder only: DOCCI +26% |
| Pre-trained encoder + Recaptioned | 51.9 | 91.6 | Combination of both: DOCCI +71% |
| + Sentence Sampling | 56.3 | 93.0 | Uniform improvement |
| + Hard Negative Fine-tuning | 54.1 | 88.1 | Substantially boosts ARO to 92%/94% |

### Key Findings

- **Recaptioning is the largest single contributor**: Simply replacing descriptions (with identical training images) brings a 41% relative improvement (DOCCI 53.5 $\rightarrow$ 75.6), proving that data quality is indeed the bottleneck.
- **COCO retrieval benchmark is saturated**: Human annotation experiments revealed that 70.2% of our model's "errors" are actually reasonable retrieval results (deemed matching by human annotators), indicating excessive redundancy/similarity between COCO captions and images.
- **Ungrounded recaptioning remains effective**: Removing the alt-text grounding information only slightly decreases performance (90.3 $\rightarrow$ 89.3), demonstrating that the foundation model's inherent visual understanding is the primary driver.
- **Hard Negatives are crucial for compositionality**: Adding hard-negative fine-tuning jumps ARO from 65%/82% to 93%/94%, but slightly hurts DOCCI performance (91.6 $\rightarrow$ 88.1), representing a trade-off.

## Highlights & Insights

- **Triumph of Minimalism**: While competing methods design complex architectures (cross-modal encoders, localization losses, multi-stage inference), this paper achieves the best results in the simplest way (modifying data + changing the encoder). This suggests that many diagnoses of "insufficient architecture" may be incorrect, and the true bottleneck lies in data quality.
- **Discovery of COCO Saturation**: This side finding might be more valuable than the main experiments. COCO has long been the standard benchmark for multimodal retrieval, yet the authors discovered that 70%+ of "errors" are misjudgments. This implies that many past minor gains on COCO may be meaningless. New benchmarks like DOCCI should become the standard.
- **Reusable Recaptioning Paradigm**: The approach of using a foundation model with grounding information to recaption training data is generalizable and can be transferred to any multimodal task to improve data quality.

## Limitations & Future Work

- Zero-shot classification on ImageNet is only 68.4%, lagging behind the SOTA of 82.6%. The authors attribute this to differences in training data distribution (CoCa uses JFT); adding JFT improves it to 79.1%, proving it is not a methodological issue but a gap remains in practice.
- The computational cost of recaptioning 1B images is massive (even though it is done only once), making it hard to replicate for resource-constrained researchers.
- Larger vision encoders were not explored (only ViT-Base was used), so the scaling effects remain unknown.
- Hard negative fine-tuning slightly harms retrieval performance, requiring a better way to balance the two.
- Evaluation was only conducted on image-text retrieval and classification; the performance on other downstream tasks (such as VQA and visual reasoning) remains unknown.

## Related Work & Insights

- **vs NegCLIP**: NegCLIP incorporates manually constructed hard negatives during training, reaching 71%/81% (ARO). The proposed method reaches 92%/94%, proving that adding negative samples alone is insufficient, and data quality (recaptioning) is the more fundamental improvement.
- **vs X-VLM/BLIP**: These two methods achieve SOTA using cross-modal encoders + localization information, but require two-stage inference. This work surpasses them without any architectural changes while maintaining the efficiency of standard ranking-based retrieval.
- **vs CapPa**: CapPa replaces the contrastive objective with a captioning objective to enhance compositionality, which changes the training paradigm. This work adheres to the contrastive learning paradigm, demonstrating that the issue lies in the data rather than the objective function.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The method itself is "enhanced data engineering," but it validates an important hypothesis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Numerous ablation experiments, evaluations across multiple benchmarks, and even human annotation verification.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear and persuasive narrative, with ablation experiments progressing in a step-by-step manner.
- **Value**: ⭐⭐⭐⭐⭐ Proves that data quality is the key bottleneck in multimodal learning, offering strong practical guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] DetailSemNet: Elevating Signature Verification through Detail-Semantic Integration](../../ECCV2024/interpretability/detailsemnet_elevating_signature_verification_through_detail-semantic_integratio.md)
- [\[NeurIPS 2025\] Time-Evolving Dynamical System for Learning Latent Representations of Mouse Visual Cortex](../../NeurIPS2025/interpretability/time-evolving_dynamical_system_for_learning_latent_representations_of_mouse_visu.md)
- [\[CVPR 2025\] Open Ad-Hoc Categorization with Contextualized Feature Learning](open_ad-hoc_categorization_with_contextualized_feature_learning.md)
- [\[CVPR 2025\] Language Guided Concept Bottleneck Models for Interpretable Continual Learning](language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)
- [\[ICCV 2025\] SVIP: Semantically Contextualized Visual Patches for Zero-Shot Learning](../../ICCV2025/interpretability/svip_semantically_contextualized_visual_patches_for_zero-shot_learning.md)

</div>

<!-- RELATED:END -->
