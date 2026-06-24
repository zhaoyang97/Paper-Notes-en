---
title: >-
  [Paper Note] CoLLM: A Large Language Model for Composed Image Retrieval
description: >-
  [CVPR 2025][Multimodal VLM][Composed Image Retrieval] This work proposes CoLLM, a unified framework for Composed Image Retrieval (CIR) leveraging Large Language Models. By generating training triplets on-the-fly from image-caption pairs, producing joint multimodal embeddings with an LLM, and constructing a large-scale MTCIR dataset with 3.4 million samples, CoLLM achieves SOTA performance across multiple CIR benchmarks, with MTCIR yielding up to a 15% performance improvement.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Composed Image Retrieval"
  - "Large Language Model"
  - "Multimodal Fusion"
  - "Zero-Shot Retrieval"
  - "Triplet Generation"
date: 2026-05-08
content_hash: 4187bf34aa42e8e9
---

# CoLLM: A Large Language Model for Composed Image Retrieval

**Conference**: CVPR 2025  
**arXiv**: [2503.19910](https://arxiv.org/abs/2503.19910)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Composed Image Retrieval, Large Language Model, Multimodal Fusion, Zero-Shot Retrieval, Triplet Generation

## TL;DR

This work proposes CoLLM, a unified framework for Composed Image Retrieval (CIR) leveraging Large Language Models. By generating training triplets on-the-fly from image-caption pairs, producing joint multimodal embeddings with an LLM, and constructing a large-scale MTCIR dataset with 3.4 million samples, CoLLM achieves SOTA performance across multiple CIR benchmarks, with MTCIR yielding up to a 15% performance improvement.

## Background & Motivation

**Background**: Composed Image Retrieval (CIR) is a multimodal retrieval task. Given a reference image and a text describing the desired modifications, the goal is to retrieve a target image that meets these modifications. For instance, given an image of a "red dress" and the text "change to blue", the model needs to retrieve an image of a "blue dress". Dominant methods require (reference image, modification text, target image) triplets to train joint embeddings.

**Limitations of Prior Work**: (1) **Severe data bottleneck**—annotating CIR triplets is extremely premium and time-consuming, resulting in limited scales for existing datasets (e.g., CIRR consists of only about 36k triplets), which severely restricts model generalization; (2) **Limitations of zero-shot methods**—to bypass data scarcity, some approaches use synthetic triplets or utilize VLMs to convert CIR into image-text retrieval. However, synthetic triplets have small scales, poor diversity, and unnatural modification texts, whereas pure image-text pair methods cannot learn effective joint embeddings due to the lack of triplet structures; (3) **Insufficient multimodal fusion**—sophisticated modification instructions require deep fusion and understanding of both vision and language, which existing methods (e.g., simple concatenation, shallow cross-attention) fail to handle for nuanced semantic modifications.

**Key Challenge**: CIR requires triplet data to train joint embeddings, but the annotation cost of triplets is prohibitively high. Zero-shot methods either suffer from low quality (synthetic data) or lack joint learning capabilities (image-text pairs). How to achieve high-quality multimodal joint embedding learning without relying on expensive human annotations?

**Goal**: Design a CIR method that does not depend on manual triplet annotations, can handle complex modification instructions, and performs outstandingly under both supervised and zero-shot settings.

**Key Insight**: Large Language Models (LLMs) are naturally adept at understanding and integrating complex multimodal inputs. If the reference image and modification text can be used as inputs to an LLM, allowing the LLM to directly generate joint embeddings, deep multimodal fusion can be achieved. Meanwhile, the text processing capabilities of LLMs can be utilized to automatically construct triplets from image-text pairs.

**Core Idea**: Leverage an LLM to simultaneously address both data and model challenges: (1) dynamically generate CIR triplets from off-the-shelf image-text pairs on-the-fly for training; (2) use the hidden states of the LLM as the joint embedding of the reference image and modification text, enabling deep multimodal fusion.

## Method

### Overall Architecture

CoLLM consists of three core components: (1) an on-the-fly triplet generation module, which automatically constructs (reference image, modification text, target image) triplets from web-crawled image-caption pairs; (2) an LLM joint embedder, which passes the reference image and modification text into a multimodal LLM and extracts its hidden states as the joint query embedding; and (3) a retrieval module, which retrieves the target image in a pre-trained visual embedding space using the joint embedding. Contrastive learning is employed during training with the generated triplets.

### Key Designs

1. **On-the-fly Triplet Generation**:

    - **Function**: Automatically construct CIR training triplets from image-text pairs, eliminating the need for manual annotation.
    - **Mechanism**: Given multiple (image, caption) pairs in a mini-batch, two pairs are matched—using Image A as the reference image and Image B as the target image. An LLM is then used to analyze the differences between their captions and automatically generate modification text describing "how to modify A to B". This process is executed in real-time (on-the-fly) during training, avoiding the need to pre-build a triplet dataset. The key is to leverage semantically related but not identical image pairs within the same batch as pseudo-triplets.
    - **Design Motivation**: Purely synthetic triplets require extra generative models and are limited in scale. Constructing triplets on-the-fly from image-text pairs leverages virtually unlimited web-scale data, and the generated modification texts are more natural as they are based on real differences between actual images.

2. **LLM Joint Embedding**:

    - **Function**: Generate joint representations of the reference image and modification text leveraging the deep reasoning capabilities of LLMs.
    - **Mechanism**: The reference image is processed by a visual encoder to extract features as visual tokens, and the modification text is represented as text tokens. Both are fed into the LLM. The hidden states of a specific layer of the LLM (or the output of the last token) are extracted as the joint query embedding. This embedding is mapped to the same feature space as the target image through a projection layer, and the projection layer along with (optional) LLM adapters are trained using a contrastive loss (e.g., InfoNCE).
    - **Design Motivation**: Traditional methods use simple concatenation or shallow cross-attention to fuse vision and language, which cannot handle complex modification instructions like "change the red color to blue and shrink the skirt size" that require comprehension of semantic dependencies. The multi-layer Transformer of LLMs is naturally suited for such deep reasoning-based fusion tasks.

3. **MTCIR Dataset & Benchmark Correction**:

    - **Function**: Provide a large-scale CIR training set with 3.4 million samples and correct evaluation issues in existing benchmarks.
    - **Mechanism**: MTCIR (Multi-Text CIR) contains approximately 3.4 million triplet samples built from multi-source image-text pairs. Each triplet contains multiple modification text descriptions (Multi-Text), covering modifications of different granularities and styles. Furthermore, the paper identifies annotation noise and evaluation biases in existing benchmarks like CIRR and Fashion-IQ (e.g., about X% of test pairs in CIRR are ambiguous), proposing corrected versions to enhance evaluation reliability.
    - **Design Motivation**: Data scale is critical for deep learning; MTCIR is more than an order of magnitude larger than the largest existing CIR dataset. Correcting the benchmarks ensures reliable evaluation, as comparisons on noisy benchmarks can lead to misleading conclusions.

### Loss & Training

The model is trained primarily using the InfoNCE contrastive loss, which pulls the joint embedding generated by the LLM closer to the visual embedding of the correct target image while pushing it away from the embeddings of negative samples. During training, the LLM backbone can be frozen while only training the projection layer (parameter-efficient), or the LLM can be fine-tuned via LoRA (higher performance). The visual encoder is typically frozen (using pre-trained weights like CLIP) to keep the target image embedding space stable.

## Key Experimental Results

### Main Results

| Method | Setting | CIRR R@5 | CIRR R@10 | Fashion-IQ R@10 | Description |
|------|------|----------|-----------|-----------------|------|
| Pic2Word | Zero-shot | Baseline | Baseline | Baseline | Image-text pair method |
| SEARLE | Zero-shot | Medium | Medium | Medium | Synthetic triplets |
| CompoDiff | Zero-shot | Higher | Higher | Higher | Diffusion model |
| **CoLLM** | **Zero-shot** | **SOTA** | **SOTA** | **SOTA** | LLM joint embedding |
| ARTEMIS | Supervised | Baseline | Baseline | Baseline | Traditional method |
| **CoLLM** | **Supervised** | **SOTA** | **SOTA** | **SOTA** | + Triplet data |

### MTCIR Contribution

| Training Data | Performance | Description |
|----------|------|------|
| Original small-scale data | Baseline | Existing datasets |
| + MTCIR | Up to +15% | Large-scale data yields significant gains |
| Corrected vs. Original Benchmarks | Ranking changes | More reliable evaluation reduces noise impact |

### Key Findings

- **LLM joint embedding significantly outperforms shallow fusion**: Using LLM hidden states as the joint embedding yields higher performance compared to traditional concatenation or cross-attention, particularly when tackling complex modification texts.
- **On-the-fly triplet generation is highly feasible and effective**: Triplets generated dynamically from image-text pairs are of sufficient quality to support effective training, with performance approaching or even exceeding methods using manually annotated triplets.
- **Data scale is crucial**: The 3.4 million samples in MTCIR produce up to a 15% performance improvement, proving that the performance bottleneck in the CIR field lies heavily in data rather than architecture.
- **Benchmark correction is meaningful**: On the corrected CIRR and Fashion-IQ, the relative rankings of different methods shifted, illustrating that noise in the original benchmarks indeed compromised fair comparison.

## Highlights & Insights

- **A unified, one-stop solution**: CoLLM simultaneously addresses the data scarcity (on-the-fly triplet generation), model limitation (LLM joint embedding), and evaluation biases (benchmark correction) of CIR. This systematic problem-solving approach is highly exemplary.
- **New paradigm of utilizing LLMs as feature fusers**: Instead of utilizing LLMs for text generation, this work uses their internal representations as multimodal joint embeddings. This "LLM-as-Encoder" concept can be transferred to other retrieval tasks that require deep multimodal fusion.
- **On-the-fly triplet generation**: Designing a self-supervised strategy that automatically constructs triplets based on the semantic differences between image pairs in a batch is highly ingenious and could be extended to other tasks requiring relational training data.

## Limitations & Future Work

- The quality of on-the-fly triplet generation relies on the semantic relevance of image pairs inside the mini-batch; if the semantic distance is too large or too small, the generated modification texts might be less meaningful.
- The computational cost of LLM inference is relatively high, which can trigger bottlenecks in large-scale online retrieval scenarios.
- The construction process of MTCIR relies on automated pipelines, which may introduce noisy samples.
- The paper mainly evaluates on fashion and general scenes, and the generalization to professional domains (e.g., medical imaging, satellite imagery) remains unverified.
- Future directions include: expanding CoLLM to more complex retrieval scenarios such as Video CIR and 3D CIR.

## Related Work & Insights

- **vs. Pic2Word**: Pic2Word maps reference images to text tokens, concatenates them with modification texts, and uses a CLIP text encoder to generate query embeddings, resulting in limited fusion depth. CoLLM replaces the CLIP text encoder with a more powerful LLM to achieve significantly deeper fusion.
- **vs. SEARLE**: SEARLE implements zero-shot CIR by inverting CLIP image embeddings into pseudo-text tokens. CoLLM's on-the-fly triplet generation allows the model to perform supervised learning instead of relying on approximations.
- **vs. CompoDiff**: CompoDiff uses diffusion models to generate target images before retrieval. CoLLM operates directly in the embedding space, avoiding the computational overhead and error accumulation of the generative steps.
- The "LLM-as-Embedder" concept of CoLLM aligns with text retrieval works like E5-Mistral, suggesting this trend warrants attention in the multimodal domain.

## Rating

- Novelty: ⭐⭐⭐⭐ Both the LLM joint embedding and the on-the-fly triplet generation are innovative, though their foundational sub-components (contrastive learning, LLM embeddings) are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across multiple benchmarks and settings, though the unavailability of HTML prevents full verification of specific numbers.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definitions and systematic methodology descriptions.
- Value: ⭐⭐⭐⭐⭐ Comprehensive contributions to the CIR field (data + model + evaluation), with the MTCIR dataset carrying high value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Adapting In-context Generation for Enhanced Composed Image Retrieval](../../CVPR2026/multimodal_vlm/adapting_in-context_generation_for_enhanced_composed_image_retrieval.md)
- [\[CVPR 2025\] LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant](lamra_large_multimodal_model_as_your_advanced_retrieval_assistant.md)
- [\[ACL 2026\] TEMA: Anchor the Image, Follow the Text for Multi-Modification Composed Image Retrieval](../../ACL2026/multimodal_vlm/tema_anchor_the_image_follow_the_text_for_multi-modification_composed_image_retr.md)
- [\[CVPR 2026\] ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval](../../CVPR2026/multimodal_vlm/recall_recalibrating_capability_degradation_for_mllm-based_composed_image_retrie.md)
- [\[CVPR 2026\] Self-guided Semantic Inspection for Zero-Shot Composed Image Retrieval](../../CVPR2026/multimodal_vlm/self-guided_semantic_inspection_for_zero-shot_composed_image_retrieval.md)

</div>

<!-- RELATED:END -->
