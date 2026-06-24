---
title: >-
  [Paper Note] Open Ad-Hoc Categorization with Contextualized Feature Learning
description: >-
  [CVPR 2025][Interpretability][Ad-Hoc Categorization] This paper proposes OAK (Open Ad-hoc Categorization with Contextualized Feature Learning). By introducing a few learnable context tokens into the input layer of a frozen CLIP model and combining CLIP's vision-language alignment objective with the visual clustering objective of GCD, the method achieves adaptive ad-hoc category discovery and context switching under few-labeled samples. It achieves an accuracy of 87.4% on nove…
tags:
  - "CVPR 2025"
  - "Interpretability"
  - "Ad-Hoc Categorization"
  - "Contextualized Learning"
  - "CLIP"
  - "Generalized Category Discovery"
date: 2026-05-08
content_hash: fa09a816f82b0223
---

# Open Ad-Hoc Categorization with Contextualized Feature Learning

**Conference**: CVPR 2025  
**arXiv**: [2512.16202](https://arxiv.org/abs/2512.16202)  
**Code**: [https://github.com/Wayne2Wang/OAK](https://github.com/Wayne2Wang/OAK)  
**Area**: Interpretability  
**Keywords**: Ad-Hoc Categorization, Contextualized Learning, CLIP, Generalized Category Discovery, Interpretability

## TL;DR

This paper proposes OAK (Open Ad-hoc Categorization with Contextualized Feature Learning). By introducing a few learnable context tokens into the input layer of a frozen CLIP model and combining CLIP's vision-language alignment objective with the visual clustering objective of GCD, the method achieves adaptive ad-hoc category discovery and context switching under few-labeled samples. It achieves an accuracy of 87.4% on novel classes of the Stanford Mood dataset, outperforming CLIP and GCD by over 50%.

## Background & Motivation

**Background**: Traditional visual classification assumes a fixed universe of generic categories (e.g., plants, animals). However, real-world scenarios often require ad-hoc (temporary) categorization—such as "things that can be sold at a garage sale." These categories lack visual or semantic similarity and are dynamically created for specific goals. Open-vocabulary classification models like CLIP achieve flexible classification via vision-language alignment, but they rely on a fixed global semantic space and cannot adapt to different contexts. GCD (Generalized Category Discovery) discovers novel categories through visual clustering but lacks semantic guidance.

**Limitations of Prior Work**: The same image should be classified into completely different categories under different contexts (e.g., "drinking water" belongs to the Action category, "residential area" belongs to Location, and "focused" belongs to Mood). However, existing methods only provide a single, fixed explanation. CLIP's image encoder does not adjust its attention based on context (always focusing on salient objects), while GCD lacks semantic clues and easily fails on complex ad-hoc categories.

**Key Challenge**: Ad-hoc categorization relies on the same perceptual mechanisms as generic classification but additionally requires contextualization to adapt to different goals—how to allow the model to dynamically adjust feature representations based on context while retaining generic perceptual capabilities?

**Goal**: Propose the open ad-hoc categorization task: given a few labeled samples and a large volume of unlabeled data, the model needs to (1) infer the latent context, and (2) expand ad-hoc categories through semantic expansion and visual clustering.

**Key Insight**: Inspired by cognitive science—humans identify ad-hoc categories using the same perceptual mechanisms as generic categories but require contextualization to adapt to different goals. Therefore, instead of modifying CLIP's perceptual mechanism, this work captures the implicit context semantics in the data through learnable context tokens.

**Core Idea**: Inject a few learnable context tokens into the input layer of a frozen CLIP model, co-training them with the GCD clustering objective and the CLIP text-guided objective to achieve context-aware feature modulation and ad-hoc category discovery.

## Method

### Overall Architecture

OAK is based on a frozen CLIP ViT image encoder. For each context, it independently learns a set of context tokens $\mathbf{z}_c$ and feeds them into the ViT alongside image patch tokens: $f_c(\mathbf{x}_i) := f([\mathbf{x}_i, \mathbf{z}_c])$. Training utilizes a joint objective: GCD's contrastive learning clustering objective + CLIP's image-text classification objective. During inference, classification results under different contexts can be obtained simply by switching the context tokens.

### Key Designs

1. **Context-aware Visual Attention**:

    - **Function**: Dynamically adjusts image features based on context, guiding the encoder to focus on relevant regions.
    - **Mechanism**: Learn a set of context tokens for each context as extra input tokens to the ViT. These tokens are similar to register tokens but are optimized independently for each context while keeping the backbone frozen. Through the self-attention mechanism, the presence of context tokens alters the attention patterns—e.g., focusing on hands under the Action context, background under Location, and facial expressions under Mood.
    - **Design Motivation**: Different contexts require focusing on different regions of the image. By injecting learnable context signals at the input layer, feature space contextualization is achieved with minimal parameter overhead, while fully retaining the pre-trained perceptual capability of CLIP.

2. **Bottom-up Visual Clustering**:

    - **Function**: Discovers novel categories by clustering visual features.
    - **Mechanism**: Adopt the GCD contrastive learning framework, utilizing a self-supervised contrastive loss $\ell_{\text{self-con}}$ for unlabeled data and a supervised contrastive loss $\ell_{\text{sup-con}}$ for labeled data, optimized jointly: $\ell_{\text{GCD}}(\mathbf{z}) = (1-\lambda)\ell_{\text{self-con}}(\mathbf{z}; \mathcal{D_U}) + \lambda\ell_{\text{sup-con}}(\mathbf{z}; \mathcal{D_L})$. This objective only optimizes the context tokens.
    - **Design Motivation**: Pure text guidance might miss visually distinguishable but semantically non-obvious novel categories; visual clustering serves as a complementary mechanism to discover novel category structures.

3. **Top-down Text Guidance**:

    - **Function**: Leverages CLIP's semantic knowledge to guide clustering alignment with semantic categories.
    - **Mechanism**: Freeze the text encoder $g$, and construct a classification loss over known classes $\mathcal{Y}_\mathcal{L}$ and potential novel classes $\hat{\mathcal{Y}}_\mathcal{N}$ generated by LLMs. Ground-truth labels are used for labeled data, while pseudo-labels for unlabeled data are generated via SS-KMeans + Hungarian matching. The overall objective is $\ell_{\text{OAK}}(\mathbf{z}_c) = \ell_{\text{GCD}}(\mathbf{z}_c) + \lambda_{\text{text}} \cdot \ell_{\text{text}}(\mathbf{z}_c)$.
    - **Design Motivation**: The GCD objective treats categories as independent entities, ignoring semantic relationships. Text guidance utilizes CLIP's rich semantic knowledge to align visual clustering with meaningful semantic labels.

### Loss & Training

- The overall objective is a weighted sum of the GCD contrastive loss and the text-guided classification loss.
- Pseudo-labels (updated every epoch) are used for unlabeled data in the text guidance loss, obtained via semi-supervised K-means + Hungarian matching.
- Only the context tokens are optimized; the CLIP backbone and text encoder are completely frozen.
- A set of context tokens is trained independently for each context.

## Key Experimental Results

### Main Results

Stanford dataset overall accuracy (Overall):

| Method | Action | Location | Mood | Omni |
|------|--------|----------|------|------|
| CLIP-ZS + LLM vocab | 65.2 | 47.5 | 55.0 | 43.0 |
| CLIP-ZS + GT vocab | 86.7 | 59.7 | 72.1 | 38.3 |
| GCD | 78.3 | 77.8 | 52.1 | 52.3 |
| **OAK** | **86.9** | **85.9** | **78.4** | **70.3** |

Stanford Novel class accuracy:

| Method | Action | Location | Mood |
|------|--------|----------|------|
| CLIP-ZS + LLM vocab | 38.6 | 34.2 | 35.4 |
| GCD | 67.8 | 80.8 | 40.6 |
| **OAK** | **85.1** | **88.4** | **87.4** |

OAK significantly outperforms CLIP (35.4%) and GCD (40.6%) on Mood novel classes with an accuracy of 87.4%.

### Ablation Study

Validation on the Clevr-4 dataset demonstrates that OAK is also effective on synthetic data: OAK achieves a novel class accuracy of 47.8% on Texture (compared to GCD's 43.6%) and reaches 100% on Color.

### Key Findings

- Saliency maps of OAK display clear context switching: Action focuses on hands, Location focuses on the background, and Mood focuses on facial expressions, aligning closely with human intuition.
- In terms of Omni accuracy (correct prediction consistently across all contexts), OAK (70.3%) far exceeds the baselines (GCD 52.3%, CLIP 43.0%), indicating outstanding context consistency.
- Text guidance is particularly helpful for concepts that CLIP is less familiar with (e.g., Location, Mood).

## Highlights & Insights

- Proposes a novel and cognitively grounded task definition—open ad-hoc categorization, which is highly practical.
- Method design is extremely simple: it only requires adding a few learnable tokens to the CLIP input layer, without modifying any model architecture.
- Saliency map results are highly convincing, visually demonstrating the capability of context switching.
- The introduction of the Omni accuracy metric is valuable for assessing the model's ability to switch seamlessly across multiple contexts.
- Organically fuses the strengths of GCD and CLIP with high complementarity.

## Limitations & Future Work

- The discovery of novel category names relies on LLM prompting, and the quality is restricted by the limitations of the LLM.
- Currently, context tokens are trained independently for each context; knowledge sharing across contexts has not yet been explored.
- The scale of the Stanford dataset is relatively small; performance in large-scale real-world scenarios remains to be validated.
- Automatically discovering the contexts themselves (rather than being given the category names for a specified context) remains a more challenging open problem.

## Related Work & Insights

- Compared to LLM/VLM-based multiple clustering methods like IC||TC and OpenSMC, OAK is more efficient as it does not rely on external captions.
- GCD (Generalized Category Discovery) provides the foundation for visual clustering, which OAK extends with contextualization capabilities.
- Visual prompt tuning (VPT) provides the technical foundation for context tokens, but OAK extends it from adaptation to a single task to adaptation to multiple contexts.
- Provides important insights for the fields of few-shot visual classification and contextualized representation learning.

## Rating

- **Novelty**: 9/10 — Novel task definition, simple and elegant method.
- **Experimental Thoroughness**: 8/10 — Validated across multiple datasets and metrics, with excellent saliency map analysis.
- **Writing Quality**: 9/10 — Clear logic, with in-depth elaboration on the connection to cognitive science.
- **Value**: 8/10 — Key pioneer of a new direction in ad-hoc categorization, with high method scalability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SVIP: Semantically Contextualized Visual Patches for Zero-Shot Learning](../../ICCV2025/interpretability/svip_semantically_contextualized_visual_patches_for_zero-shot_learning.md)
- [\[CVPR 2025\] Language Guided Concept Bottleneck Models for Interpretable Continual Learning](language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)
- [\[CVPR 2025\] Learning Visual Composition through Improved Semantic Guidance](learning_visual_composition_through_improved_semantic_guidance.md)
- [\[NeurIPS 2025\] Evaluating LLMs in Open-Source Games](../../NeurIPS2025/interpretability/evaluating_llms_in_open-source_games.md)
- [\[AAAI 2026\] Quiet Feature Learning in Algorithmic Tasks](../../AAAI2026/interpretability/quiet_feature_learning_in_algorithmic_tasks.md)

</div>

<!-- RELATED:END -->
