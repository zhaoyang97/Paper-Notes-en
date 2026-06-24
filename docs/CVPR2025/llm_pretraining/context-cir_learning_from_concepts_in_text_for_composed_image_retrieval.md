---
title: >-
  [Paper Note] ConText-CIR: Learning from Concepts in Text for Composed Image Retrieval
description: >-
  [CVPR 2025][LLM Pretraining][Composed Image Retrieval] The ConText-CIR framework is proposed, which utilizes a Text Concept-Consistency loss to align noun phrases in text modifications with corresponding regions in the query image. Combined with a synthetic data generation pipeline, it achieves SOTA performance on multiple CIR benchmarks.
tags:
  - "CVPR 2025"
  - "LLM Pretraining"
  - "Composed Image Retrieval"
  - "Text Concept-Consistency"
  - "CLIP"
  - "zero-shot retrieval"
  - "synthetic data"
date: 2026-05-08
content_hash: 2ccd22ee2974a4c7
---

# ConText-CIR: Learning from Concepts in Text for Composed Image Retrieval

**Conference**: CVPR 2025  
**arXiv**: [2505.20764](https://arxiv.org/abs/2505.20764)  
**Code**: [mvrl/ConText-CIR](https://github.com/mvrl/ConText-CIR)  
**Institution**: Washington University in St. Louis / Saint Louis University / George Washington University  
**Area**: Image Retrieval / Vision-Language Models  
**Keywords**: Composed Image Retrieval, Text Concept-Consistency, CLIP, zero-shot retrieval, synthetic data

## TL;DR
The ConText-CIR framework is proposed, which utilizes a Text Concept-Consistency loss to align noun phrases in text modifications with corresponding regions in the query image. Combined with a synthetic data generation pipeline, it achieves SOTA performance on multiple CIR benchmarks.

## Background & Motivation

**Background**: Composed Image Retrieval (CIR) is a multi-modal retrieval task where users provide both a query image and a text modification description, and the model retrieves target images that satisfy the modification conditions. Combining the advantages of image and text retrieval, this task is widely applied in scenarios such as visual search and e-commerce recommendation.

**Limitations of Prior Work**:
   - Existing CIR methods struggle to accurately represent the relationship between images and text modifications, leading to suboptimal performance.
   - When the text contains multiple semantic conditions, models often fail to satisfy all conditions simultaneously (as shown in the failure cases in Fig. 1).
   - Image embeddings are highly complex and difficult to encode with specific retrieval conditions, while pure text is inadequate for precisely describing complex visual information.

**Key Challenge**: The correspondence between concepts (noun phrases) in text modifications and the query image lacks explicit supervision, making it difficult for the model to learn "which text concepts should focus on which parts of the image."

**Key Insight**: Introduce concept-level representation learning to align the representations of individual noun phrases in the text with the corresponding regions in the query image.

**Core Idea**: Text Concept-Consistency Loss + Synthetic Data Pipeline = Concept-Level Aligned Composed Image Retrieval.

## Method

### Overall Architecture
ConText-CIR is built upon the CLIP vision-language model and consists of three core components: feature extraction, concept-consistency learning, and retrieval inference.

### Key Designs

1. **Text Concept-Consistency Loss (TCC Loss)**

    - **Function**: Encourage the representations of noun phrases in text modifications to remain consistent with the representations of related regions in the query image.
    - **Mechanism**: Extract noun phrases from the text modifications, compute the attention distribution between each noun phrase and the query image patch tokens, and enforce alignment between the conceptual representation of the noun phrase and the corresponding region in the image.
    - **Design Motivation**: Previous methods only aligned text and images at the global level, ignoring fine-grained correspondence at the concept level.
    - **Effect**: Enable the model to simultaneously focus on multiple semantic conditions in the text.

2. **Synthetic Data Generation Pipeline**

    - **Function**: Automatically generate training data from existing CIR datasets or unlabeled images.
    - **Mechanism**: Utilize vision-language models to generate image captions, and employ LLMs to generate modification texts, constructing (query image, text modification, target image) triplets.
    - **Advantages**: No increase in inference time complexity, and no requirement for large-scale additional annotated data.
    - **Supports scaling and generating the CIRRR dataset from the CIRR dataset.**

3. **Inference Strategy**

    - Combine the embeddings of the query image and text modification to perform nearest neighbor retrieval in the target image database.
    - No increase in inference time complexity.
    - Supports multiple CLIP backbones (ViT-B, ViT-L, ViT-H).

### Loss & Training
- Fine-tuned based on pretrained CLIP models.
- Jointly trained using TCC Loss and standard CIR loss.
- Jointly trained with a mixture of synthetic and real data.

## Key Experimental Results

### CIRR Supervised Setting (Recall@K)

| Method | Backbone | R@1 | R@5 | R@10 | R@50 |
|------|----------|------|------|-------|-------|
| CLIP4CIR | ViT-B | 44.82 | 77.04 | 86.65 | 97.90 |
| CASE | ViT-B | 48.68 | 79.98 | 88.51 | — |
| **ConText-CIR** | **ViT-B** | **Best** | **Best** | **Best** | **Best** |

### CIRR Zero-Shot Setting (R@1 Gain)

| Backbone | R@1 Gain (vs. Prev. SOTA) |
|----------|---------------------------|
| ViT-B | +4.78 |
| ViT-L | +5.38 |
| ViT-H | +12.88 |

**Key Findings**: ConText-CIR with a ViT-H backbone even outperforms all methods using the larger ViT-G backbone (where ViT-G has approximately 400M more parameters), achieving zero-shot performance superior to CoVR-2 which was pretrained on 4.9 million samples.

### Ablation Study

| Training Data | R@1 | R@5 | R@10 | R@50 |
|---------|------|------|-------|-------|
| CIRR only | 45.25 | 77.52 | 86.88 | 97.24 |
| CIRR + CIRRR (synthetic) | 48.54 | — | — | — |

Both TCC Loss and synthetic data contribute significantly to performance.

## Technical Details Supplement

### Noun Phrase Extraction
- spaCy is used to automatically extract noun phrases from text modifications.
- Attention weights between each noun phrase and image patches are computed independently.
- The alignment loss is imposed at the token level, which is much finer-grained than the sentence level.

### CIRRR Synthetic Dataset
- Starting from original CIRR images, VLMs are utilized to generate descriptions.
- LLMs automatically generate relative modification texts based on description differences.
- The quality of synthetic data is validated through human evaluation.
- The data volume is moderate to avoid introducing noisy labels.

## Highlights & Insights
- The **concept-level alignment** idea is elegant and effective, showing a natural and meaningful progression from global alignment to fine-grained conceptual alignment.
- The **synthetic data pipeline** has strong generalizability, enabling the generation of CIR training data from any unlabeled images.
- No computational overhead is added during inference; concept consistency is introduced only during the training phase.
- Outperforming larger models with a smaller model in the zero-shot setting demonstrates the data efficiency of the method.
- Supports multiple CLIP backbones with consistent performance gains, showing excellent generalizability of the method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Atlas of In-Context Learning: How Attention Heads Shape In-Context Retrieval Augmentation](../../NeurIPS2025/llm_pretraining/the_atlas_of_in-context_learning_how_attention_heads_shape_in-context_retrieval_.md)
- [\[CVPR 2025\] DreamText: High Fidelity Scene Text Synthesis](dreamtext_high_fidelity_scene_text_synthesis.md)
- [\[CVPR 2025\] AMO Sampler: Enhancing Text Rendering with Overshooting](amo_sampler_enhancing_text_rendering_with_overshooting.md)
- [\[CVPR 2025\] MR-PLIP: Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation](mr_plip_multi_resolution_pathology.md)
- [\[ICML 2025\] Revisiting Continuity of Image Tokens for Cross-Domain Few-Shot Learning](../../ICML2025/llm_pretraining/revisiting_continuity_of_image_tokens_for_cross-domain_few-shot_learning.md)

</div>

<!-- RELATED:END -->
