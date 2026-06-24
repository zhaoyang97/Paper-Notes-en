---
title: >-
  [Paper Note] StoryGPT-V: Large Language Models as Consistent Story Visualizers
description: >-
  [CVPR 2025][Segmentation][Story Visualization] This paper proposes StoryGPT-V, which achieves accurate, high-quality, and temporally consistent character image generation in story visualization with low memory overhead through a two-stage training scheme: first training a Character-Aware Latent Diffusion Model (Char-LDM) for high-quality character generation, and then aligning LLM output with the input space of Char-LDM to achieve anaphora resolution and contextual consistenc…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Story Visualization"
  - "Large Language Models"
  - "Latent Diffusion Models"
  - "Anaphora Resolution"
  - "Character Consistency"
date: 2026-05-08
content_hash: f56fbc817cb54a35
---

# StoryGPT-V: Large Language Models as Consistent Story Visualizers

**Conference**: CVPR 2025  
**arXiv**: [2312.02252](https://arxiv.org/abs/2312.02252)  
**Code**: None  
**Area**: Image Segmentation  
**Keywords**: Story Visualization, Large Language Models, Latent Diffusion Models, Anaphora Resolution, Character Consistency

## TL;DR

This paper proposes StoryGPT-V, which achieves accurate, high-quality, and temporally consistent character image generation in story visualization with low memory overhead through a two-stage training scheme: first training a Character-Aware Latent Diffusion Model (Char-LDM) for high-quality character generation, and then aligning LLM output with the input space of Char-LDM to achieve anaphora resolution and contextual consistency.

## Background & Motivation

Story Visualization is a more complex task than single-image generation: it requires generating multi-frame images based on a sequence of narrative descriptions while maintaining character and background consistency across frames. This task faces two core challenges:

**Challenge 1: Character Generation Quality**. Although existing text-to-image models can generate high-quality single-frame images, they struggle to precisely generate specific characters in story visualization. Pure text prompts are insufficient to provide detailed information about character appearance.

**Challenge 2: Anaphora Resolution**. Story descriptions frequently use pronouns ("he", "she", "they"), and the model must infer the specific characters referred to by the pronouns from context. Although Story-LDM first introduced reference resolution, its attention memory module interacts in the CLIP space, losing fine-grained linguistic understanding, and requiring the maintenance of pixel-level representations of all prior frames, causing memory requirements to grow drastically with the number of frames.

The core motivation of this paper is: **leveraging the powerful reasoning capabilities of LLMs to solve the reference resolution problem, while compressing visual information into token-level representations to reduce memory overhead**. Through the causal language modeling capability of the LLM, the model can implicitly infer pronoun references from interleaved image and text contexts.

## Method

### Overall Architecture

StoryGPT-V adopts a two-stage training scheme:
- **First Stage**: Train the Character-Aware Latent Diffusion Model (Char-LDM). By fusing character visual features into text embeddings and using character segmentation masks to guide cross-attention maps, the model improves the accuracy and fidelity of character generation.
- **Second Stage**: Align the output of the LLM (OPT-6.7B or Llama2-7B) with the input space of Char-LDM, enabling the LLM to receive interleaved image-text inputs and produce visual outputs, leveraging causal language modeling to achieve reference resolution.

### Key Designs

1. **Character-Augmented Fused Embedding**:
    - **Function**: Fuses character visual features into text embeddings to provide character appearance information for the diffusion model.
    - **Mechanism**: For each character name token in the text description, its CLIP text embedding is concatenated with the CLIP visual embedding of the corresponding character image, and then mapped through an MLP: $c^k = \text{MLP}(\text{concat}(\psi(S[i_c^k]), \phi(I_c^k)))$
    - **Design Motivation**: Pure text descriptions (e.g., "Fred") cannot convey details of the character's visual appearance. After fusing visual features, the diffusion model can obtain richer character information during the denoising process.

2. **Cross-Attention Control with Segmentation Masks**:
    - **Function**: Guides the attention of character tokens in the diffusion model to focus on the spatial regions of the corresponding characters.
    - **Mechanism**: Uses SAM to obtain the character segmentation mask $M_k$, and designs a regularization loss $\mathcal{L}_{reg} = \frac{1}{K}\sum_{k=1}^K (A_k^- - A_k^+)$, where $A_k^+$ is the average attention inside the character region and $A_k^-$ is the average attention outside the character region.
    - **Design Motivation**: In standard LDMs, a pixel can interact unconditionally with all text tokens, leading to imprecise character generation positions and appearances. Mask guidance forces character tokens to primarily influence their corresponding spatial regions.

3. **LLM Alignment for Reference Resolution**:
    - **Function**: Leverages the reasoning capabilities of LLMs to implicitly parse pronoun references and generate visual outputs aligned with the input space of Char-LDM.
    - **Mechanism**: The LLM receives interleaved $(I_1, S_1, \dots, I_{n-1}, S_{n-1}, S_n)$ as input, where images are encoded by CLIP and mapped to 4 token embeddings. The LLM generates $R$ [IMG] tokens, which are projected into the input space of Char-LDM via a Transformer-based Mapper. The training target is the alignment loss $\mathcal{L}_{align} = \|\text{Mapper}_{LDM}(h_{[IMG_{1:R}]}, q_1,\dots, q_L) - c_i\|_2^2$, where $c_i$ is the fused embedding of the non-anaphoric text.
    - **Design Motivation**: Through causal modeling and context memory, the LLM infers "Fred and Wilma are talking" from "They are talking", and generates visual conditions containing correct character information.

### Loss & Training

**First Stage**:
- Standard diffusion loss + attention regularization loss $\mathcal{L}_{reg}$
- Training strategy: 10% unconditional training (classifier-free guidance), 10% text-only training, 80% character-augmented fusion training
- Freeze the CLIP text encoder, fine-tune other modules, 25k steps, lr=1e-5

**Second Stage**:
- Token generation loss $\mathcal{L}_{gen}$ (NLL of the [IMG] token) + alignment loss $\mathcal{L}_{align}$
- Use OPT-6.7B or Llama2-7B, freeze the LLM backbone, and train the Mapper and additional token embeddings
- Precompute non-anaphoric fused embeddings as alignment targets

## Key Experimental Results

### Main Results

FlintstonesSV dataset (with anaphoric text):

| Model | Char-Acc↑ | Char-F1↑ | BG-Acc↑ | FID↓ | BLEU4↑ | CIDEr↑|
|------|-----------|----------|---------|------|--------|--------|
| StoryDALL-E | 61.83 | 78.36 | 48.10 | 44.66 | 0.4460 | 1.3373 |
| LDM | 75.37 | 87.54 | 52.57 | 32.36 | 0.4911 | 1.5103 |
| Story-LDM | 77.23 | 88.26 | 54.97 | 36.34 | 0.4585 | 1.4004 |
| **StoryGPT-V** | **87.96** | **94.17** | **56.01** | **21.71** | **0.5070** | **1.6607** |

PororoSV dataset (with anaphoric text):

| Model | Char-Acc↑ | Char-F1↑ | FID↓ | BLEU4↑ | CIDEr↑ |
|------|-----------|----------|------|--------|--------|
| StoryDALL-E | 21.03 | 50.56 | 40.39 | 0.2295 | 0.3666 |
| Story-LDM | 29.14 | 57.56 | 26.64 | 0.2420 | 0.4581 |
| **StoryGPT-V** | **36.06** | **62.70** | **19.56** | **0.2586** | **0.5279** |

### Ablation Study

First stage ablation (FlintstonesSV, without anaphoric text):

| Configuration | Char-Acc↑ | FID↓ | Description |
|------|-----------|------|------|
| w/o $\mathcal{L}_{reg}$ | 88.86 | 23.51 | Without mask guidance |
| w/o augmented text | 87.45 | 21.27 | Without character visual augmentation |
| freeze vis | 88.67 | 22.01 | Freeze visual encoder |
| Default (w/ img) | **90.36** | 21.13 | Complete Char-LDM |

Second stage ablation:

| Configuration | Char-Acc↑ | FID↓ | Description |
|------|-----------|------|------|
| Caption-Emb_text | 69.70 | 21.32 | Text input only + text embedding alignment |
| Interleave-Emb_text | 86.10 | 21.30 | Interleaved input + text embedding alignment |
| Interleave-Emb_fuse | **87.96** | 21.71 | Interleaved input + fused embedding alignment |

### Key Findings

- Interleaved image-text input (Interleave) brings a huge improvement over text-only input (Caption) (+18.26% Char-Acc), indicating that visual context is crucial for reference resolution.
- Fused embedding (Emb_fuse) further improves accuracy compared to pure text embedding (Emb_text), proving the effectiveness of fusing character visual information.
- For long-sequence (40+ frames) generation, Story-LDM runs out of memory (OOM at 42 frames on an 80G A100), whereas StoryGPT-V can exceed 50 frames.
- A stronger LLM backbone (Llama2-7B vs OPT-6.7B) brings consistent improvements (Char-Acc: 89.08% vs 87.96%).

## Highlights & Insights

- **Efficient Contextual Memory**: Compresses visual information into 4 token embeddings ($n \times 4 \times d$) instead of the pixel-level representations ($n \times h \times w \times d$) in Story-LDM, improving memory efficiency by orders of magnitude.
- **Two-Stage Decoupled Design**: The first stage focuses on character generation quality, while the second stage focuses on reference resolution, offering a clear division of labor that facilitates independent optimization.
- **Mask-Guided Attention Control**: Explicitly guides cross-attention through segmentation masks, which establishes correspondences between character tokens and spatial regions more efficiently than purely implicit learning.
- **Multimodal Generation Capability**: The model is capable of not only generating images from text but also autonomously generating text to continue the story and concurrently generating the corresponding images.
- **Consistent Human Evaluation**: MTurk evaluations consistently show that StoryGPT-V outperforms Story-LDM in visual quality, text-alignment, character accuracy, and temporal consistency.

## Limitations & Future Work

- Currently only validated on two cartoon datasets, FlintstonesSV and PororoSV, without extension to more realistic story visualization scenarios.
- Accuracy may decline in scenarios with a large number of characters or small visual differences between characters.
- Character reference images are required during inference, which limits completely open-ended story generation.
- The alignment loss in the second stage is the L2 distance, which may not be the optimal metric for semantic alignment.
- Future work could explore scaling up to larger LLMs, extending to continuous video generation, and integrating with more advanced diffusion models like SDXL.

## Related Work & Insights

- **vs Story-LDM**: Story-LDM uses an attention memory module to maintain context in the pixel space, which has high memory overhead and limited reference resolution capability; StoryGPT-V leverages LLMs to efficiently process context in the token space.
- **vs StoryDALL-E**: StoryDALL-E requires an additional source frame as input and does not handle anaphoric text; StoryGPT-V automatically resolves reference through the LLM.
- **vs Multimodal Generation Models (e.g., GILL, Emu)**: These general models are not optimized for story visualization and lack guarantees of character consistency; StoryGPT-V specifically optimizes character generation via Char-LDM.

## Rating

- Novelty: ⭐⭐⭐⭐ It represents the first attempt to combine LLM reasoning capabilities with a character-aware diffusion model for reference resolution in story visualization.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive with quantitative, qualitative, human evaluations, ablation studies, and long-sequence analyses.
- Writing Quality: ⭐⭐⭐ Generally clear, although the math equations are slightly cluttered, and some details require referring to the supplementary material.
- Value: ⭐⭐⭐⭐ Provides an efficient and practical solution for the story visualization task. The efficient contextual memory mechanism can be generalized to other sequence generation tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] F-LMM: Grounding Frozen Large Multimodal Models](f-lmm_grounding_frozen_large_multimodal_models.md)
- [\[CVPR 2025\] GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation](glus_global-local_reasoning_unified_into_a_single_large_language_model_for_video.md)
- [\[CVPR 2025\] Assessing and Learning Alignment of Unimodal Vision and Language Models (SAIL)](assessing_and_learning_alignment_of_unimodal_vision_and_language_model.md)
- [\[ECCV 2024\] VISA: Reasoning Video Object Segmentation via Large Language Models](../../ECCV2024/segmentation/visa_reasoning_video_object_segmentation_via_large_language_models.md)
- [\[CVPR 2025\] MatAnyone: Stable Video Matting with Consistent Memory Propagation](matanyone_stable_video_matting_with_consistent_memory_propagation.md)

</div>

<!-- RELATED:END -->
