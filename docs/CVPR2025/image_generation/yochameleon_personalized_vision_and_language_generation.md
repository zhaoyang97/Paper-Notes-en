---
title: >-
  [Paper Note] Yo'Chameleon: Personalized Vision and Language Generation
description: >-
  [CVPR 2025][Image Generation][Personalized generation] Yo'Chameleon is proposed to explore the personalization of Large Multimodal Models (LMMs) for the first time. Through a dual soft prompt + self-prompting mechanism along with a "soft-positive" training strategy, it achieves personalized text understanding and image generation using only 3-5 images and 32 learnable tokens.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Personalized generation"
  - "Large Multimodal Model"
  - "soft prompt"
  - "unified image-text generation"
  - "few-shot learning"
date: 2026-05-08
content_hash: 9a5832d8e51c4d56
---

# Yo'Chameleon: Personalized Vision and Language Generation

**Conference**: CVPR 2025  
**arXiv**: [2504.20998](https://arxiv.org/abs/2504.20998)  
**Code**: [https://thaoshibe.github.io/YoChameleon](https://thaoshibe.github.io/YoChameleon)  
**Area**: Multimodal VLM / Image Generation  
**Keywords**: Personalized generation, Large Multimodal Model, soft prompt, unified image-text generation, few-shot learning

## TL;DR

Yo'Chameleon is proposed to explore the personalization of Large Multimodal Models (LMMs) for the first time. Through a dual soft prompt + self-prompting mechanism along with a "soft-positive" training strategy, it achieves personalized text understanding and image generation using only 3-5 images and 32 learnable tokens.

## Background & Motivation

**Background**: Large Multimodal Models (such as GPT-4o, Chameleon) have become general-purpose AI assistants capable of understanding and generating both text and images. Personalization technologies have been widely studied in LLMs and text-to-image models—LLMs store personalized descriptions as prompts, while image generation models undergo fine-tuning using methods like DreamBooth.

**Limitations of Prior Work**: Existing LMMs are general-purpose models that lack personalized knowledge of user-specific concepts. For instance, if a user asks the model to "describe <bo> and generate a photo of <bo> reading a book in a library", and <bo> is the user's pet dog, the model cannot provide an accurate answer. Prior personalization works (such as Yo'LLaVA, MyVLM) only explore personalization for VLM text generation; extending this to the image generation modality remains unstudied.

**Key Challenge**: Two key challenges in personalization. (1) Catastrophic forgetting: Image generation tasks require fine-grained visual information, which typically requires fine-tuning the entire model (e.g., DreamBooth), but this causes the LMM to lose general knowledge. Although soft prompts can keep the model weights frozen, training with only 3-5 images fails to yield high-quality image generation. (2) Modality incompatibility: Soft prompts optimized for image understanding produce irrelevant content when used for image generation, and vice versa. Jointly training both tasks also leads to mutual degradation.

**Goal**: To achieve personalized text understanding and image generation in a single model using only 3-5 images, without undermining the general capabilities of the LMM.

**Key Insight**: The authors find that soft prompts can achieve performance close to full-model fine-tuning when ~300 real images are available. However, a user can only provide 3-5 images. The solution is to leverage visually similar "soft-positive" images to expand the training data, and use dual soft prompts to handle the two tasks separately.

**Core Idea**: Utilizing dual soft prompts (specifically for image generation and text understanding) + a self-prompting mechanism (where the model first determines the task type before choosing the prompt) + an adaptive "soft-positive" training strategy (allocating different prompt lengths based on similarity) to resolve catastrophic forgetting and modality incompatibility in LMM personalization.

## Method

### Overall Architecture

Based on the Chameleon model (with image generation capability restored via Anole), the input consists of 3-5 target concept images, and the output is a personalized text response or image generation. The core mechanism is to learn a set of trainable tokens to encode the user concept: "<sks> is <g-tokens><u-tokens>", where <g-tokens> is used for image generation and <u-tokens> is used for text understanding, totaling 32 tokens. During training, model weights are kept frozen, and only these tokens and the corresponding classifier head matrices are updated.

### Key Designs

1. **"Soft-Positive" Image Training Strategy**:

    - **Function**: Resolves the poor optimization of soft prompts caused by having only 3-5 positive samples.
    - **Mechanism**: Approximately 1000 negative images visually similar to the positive samples are retrieved from LAION-5B. They are ranked by CLIP image similarity from high to low and divided into $k-1$ groups. Key innovation: images with higher similarity are allocated more learnable tokens (i.e., longer prompts to describe more details), while the complete set of tokens is only allocated to genuine positive samples. In this way, the model learns relevant visual features from similar images while distinguishing positive samples from soft-positives through the difference in token length.
    - **Design Motivation**: Previous methods either used only 3-5 positive samples (insufficient data), relied on data augmentation (where segmentation and inpainting quality are limited), or treated all negative samples equally. The "soft-positive" method introduces a "similarity-aware" training signal, which is more effective than traditional data augmentation (improving CLIP-I from below 0.7 to 0.74) and uniform negative sample strategies.

2. **Dual Soft Prompt + Self-Prompting Mechanism**:

    - **Function**: Resolves the incompatibility of image generation and text understanding tasks when using the same set of prompts.
    - **Mechanism**: Learning two independent sets of trainable tokens—<g-tokens> (k=16 tokens for image generation) and <u-tokens> (h=16 tokens for text understanding). During training, data is structured such that the model must first predict which token set should be used for the current task (self-prompting) before executing the task. For example, for the text understanding task "<sks>是什么？", the target output first includes <u-tokens> followed by the actual answer. This forces the model to align different token sets with different tasks.
    - **Design Motivation**: Experiments demonstrate that joint training with shared tokens, simple concatenation of two token sets, and fine-tuning after concatenation are all inferior to self-prompting. The core reason is that token representations optimized for one task lack semantic relevance for the other. The elegant aspect of self-prompting is that tokens play a dual role of both "task mode selection" and "concept information encoding".

3. **Concept Representation as Learnable Prompt**:

    - **Function**: Efficiently encodes personalized concepts into a small number of trainable parameters.
    - **Mechanism**: Based on Chameleon's autoregressive training objective, the personalized concept is represented as "<sks> is <token1><token2>...<tokenk>". During training, the loss is computed only on the response part: $p(\mathbf{X}_a) = \prod_{j=1}^{L} p_{\theta}(x_j | \mathbf{X}_{a,<j})$. Trainable parameters only consist of the concept identifier <sks>, k latent tokens, and the matrices W corresponding to these tokens in the language model's final classifier head.
    - **Design Motivation**: Compared to full-model fine-tuning, the soft prompt method is computationally efficient and completely avoids catastrophic forgetting by freezing the model weights. Only 32 tokens (~0.001% of parameters) are required to achieve image generation quality close to full-model fine-tuning.

### Loss & Training

Using the standard autoregressive language modeling loss, computed only on the response part. The training data consists of two parts: (1) Understanding data—which contains identification data (positive samples + 100 easy + 100 hard negative samples) and QA data (10 template questions with answers generated by GPT-4o); (2) Generation data—positive samples + soft-positive images. Optimized using AdamW with a learning rate of 1e-4, training for 15 epochs per concept with a batch size of 4 on an A100 GPU. The best checkpoint is selected based on the mean score of identification accuracy and CLIP-I.

## Key Experimental Results

### Main Results

| Method | Token Count | Identification Accuracy↑ | QA (Vision)↑ | QA (Text)↑ | CLIP-I↑ | Face Similarity↑ |
|------|---------|------------|----------|----------|---------|-----------|
| Chameleon (Original) | 0 | 0.500 | 0.474 | 0.405 | 0.425 | 0.009 |
| Chameleon+Text Prompt | ~64 | 0.727 | 0.523 | 0.716 | 0.566 | 0.012 |
| Chameleon+Image Prompt (1k) | ~1k | 0.361 | 0.580 | 0.573 | 0.487 | 0.013 |
| GPT-4o+Text Prompt | ~64 | 0.841 | 0.923 | 0.798 | 0.636 | 0.028 |
| GPT-4o+Image Prompt (1k) | ~1k | 0.902 | 0.867 | 0.982 | 0.657 | 0.036 |
| **Yo'Chameleon (Ours)** | **32** | **0.845** | 0.604 | 0.721 | **0.783** | **0.212** |

### Ablation Study

| Training Strategy | Identification Accuracy↑ | CLIP-I↑ | Face Similarity↑ |
|---------|------------|---------|-----------|
| Shared Prompt + Language Data Only | 0.784 | 0.120 | 0.032 |
| Shared Prompt + Image Data Only (Positive Samples) | 0.104 | 0.678 | 0.188 |
| Shared Prompt + Image Data Only (Soft-Positive) | 0.108 | **0.742** | 0.225 |
| Shared Prompt + Mixed Data | 0.564 | 0.687 | 0.193 |
| Separate Prompt + Simple Concatenation | 0.502 | 0.615 | 0.156 |
| Separate Prompt + Fine-tuning after Concatenation | 0.251 | 0.648 | 0.189 |
| **Separate Prompt + Self-Prompting** | **0.747** | **0.761** | **0.224** |

### Key Findings

- **Soft-positive strategy significantly outperforms data augmentation**: Using soft-positive images yields an improvement of approximately 20% in face similarity compared to data augmentation via segmentation and inpainting, as real images offer far superior quality compared to synthetically augmented data.
- **Self-prompting is the key to balancing multi-task learning**: Jointly training both tasks with a shared prompt hurts performance in both tasks, whereas self-prompting allows the model to approach single-task optimal performance on both tasks, indicating that letting the model "first determine the task type" effectively decouples representations of different modalities.
- **Only 32 tokens outperform prompt methods with 1k+ tokens**: With only 32 learnable tokens, Yo'Chameleon substantially outperforms GPT-4o using a 1k-token image prompt in image generation (CLIP-I: 0.783 vs. 0.657), demonstrating the efficiency advantage of learned representations.
- **Room for improvement in face generation**: The current face similarity is 0.212, while a passable threshold is around 0.4. Increasing the token count can improve quality, but with diminishing returns (16 tokens is the inflection point for cost-effectiveness).

## Highlights & Insights

- **Introduction of the "soft-positive" concept**: Creatively redefines hard-negatives as varying degrees of "soft-positives" and adaptively allocates prompt lengths based on similarity. This continuous handling of positive/negative samples can be transferred to contrastive learning, retrieval-augmented generation, and other scenarios.
- **Dual role of self-prompting**: Tokens act as both task selectors and content encoders. A single set of parameters serves two functions, representing an elegant design. This concept can be extended to additional modalities (e.g., audio generation) by simply introducing new token groups and corresponding self-prompting rules.
- **Discovery that "300 images can rival full-model fine-tuning"**: This experimental insight directly inspired the soft-positive strategy. By analyzing "where the gap comes from," the essential cause of "insufficient data" was identified and addressed purposefully.

## Limitations & Future Work

- Based on the Chameleon model, its native image generation capability is weaker than specialized models like DALL-E 3, limiting the personalization performance by the capacity of the base model.
- Face similarity (0.212) remains far below passing standards (0.4); personalized portrait generation is not yet highly practical.
- Each new concept requires independent training for about 15 epochs, making instant personalization (zero-shot at inference) impossible.
- Performance on QA tasks is noticeably weaker than GPT-4o, partly due to the understanding capability gap of the base model (Chameleon) itself.
- Currently only supports single-concept personalization; multi-concept combination (e.g., "my dog in my garden") remains unresolved.

## Related Work & Insights

- **vs DreamBooth**: DreamBooth achieves high-quality personalized image generation through full-model fine-tuning but causes catastrophic forgetting. Yo'Chameleon keeps the model weights frozen via soft prompting, trading training efficiency for better preservation of general capabilities.
- **vs Yo'LLaVA**: The prior work only achieved personalization for VLM text generation. Yo'Chameleon extends this to unified text + image generation, with key additions being the soft-positive strategy and the dual prompt/self-prompting mechanism.
- **vs Textual Inversion**: Both learn tokens to represent new concepts, but Textual Inversion is used exclusively for image generation. Yo'Chameleon achieves dual personalization of both understanding and generation within a unified LMM architecture.
- **vs GPT-4o**: GPT-4o can perform basic personalization through prompt engineering, but is far inferior to learning-based methods in fine-grained visual details (particularly human faces).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to explore LMM personalization; the proposed soft-positive and self-prompting mechanisms are both novel and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Thorough ablation studies from multiple angles, but the dataset is relatively small, containing only 40 concepts.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition, rigorous motivational derivation, and highly logical experimental analysis.
- Value: ⭐⭐⭐⭐ Opens up a new direction for LMM personalization, though practical application value remains to be seen due to base model limitations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Do Visual Imaginations Improve Vision-and-Language Navigation Agents?](do_visual_imaginations_improve_vision-and-language_navigation_agents.md)
- [\[CVPR 2025\] Enhancing Vision-Language Compositional Understanding with Multimodal Synthetic Data (SPARCL)](enhancing_vision-language_compositional_understanding_with_multimodal_synthetic_.md)
- [\[CVPR 2025\] DreamCache: Finetuning-Free Lightweight Personalized Image Generation via Feature Caching](dreamcache_finetuning-free_lightweight_personalized_image_generation_via_feature.md)
- [\[CVPR 2025\] ConceptGuard: Continual Personalized Text-to-Image Generation with Forgetting and Confusion Mitigation](conceptguard_continual_personalized_text-to-image_generation_with_forgetting_and.md)
- [\[CVPR 2025\] Language-Guided Image Tokenization for Generation](language-guided_image_tokenization_for_generation.md)

</div>

<!-- RELATED:END -->
