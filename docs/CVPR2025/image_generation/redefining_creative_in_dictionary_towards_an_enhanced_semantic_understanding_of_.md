---
title: >-
  [Paper Note] Redefining <Creative> in Dictionary: Towards an Enhanced Semantic Understanding of Creative Generation
description: >-
  [CVPR 2025][Image Generation][Creative Generation] CreTok redefines "creative" as a learnable, general token `<CreTok>`. By continuously and space-iteratively optimizing the semantics of this token in the text embedding space, it endows diffusion models with "meta-creativity" for compositional creative generation. This enables the zero-shot generation of diverse concept-blended images without additional training, achieving a generation speed 10-30 times faster than current st…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Creative Generation"
  - "Textual Concept Composition"
  - "Token Redefinition"
  - "Meta-Creativity"
  - "Diffusion Models"
date: 2026-05-08
content_hash: 856547e11a193724
---

# Redefining <Creative> in Dictionary: Towards an Enhanced Semantic Understanding of Creative Generation

**Conference**: CVPR 2025  
**arXiv**: [2410.24160](https://arxiv.org/abs/2410.24160)  
**Code**: [https://github.com/fu-feng/CreTok](https://github.com/fu-feng/CreTok)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Creative Generation, Textual Concept Composition, Token Redefinition, Meta-Creativity, Diffusion Models

## TL;DR

CreTok redefines "creative" as a learnable, general token `<CreTok>`. By continuously and space-iteratively optimizing the semantics of this token in the text embedding space, it endows diffusion models with "meta-creativity" for compositional creative generation. This enables the zero-shot generation of diverse concept-blended images without additional training, achieving a generation speed 10-30 times faster than current state-of-the-art (SOTA) methods.

## Background & Motivation

**Background**: Current T2I diffusion models (e.g., SD3, DALL-E 3, Midjourney) excel at generating out-of-distribution images (e.g., "blue banana") because the semantics of "blue" are clear and concrete. However, for compositional creative generation (e.g., "a creative mixture that looks like lettuce and a mantis"), models struggle to understand the abstract concept of "creative".

**Limitations of Prior Work**: Existing creative generation methods rely on synthetic reference prompts or reference images. ConceptLab trains independent tokens for each new concept; BASS searches through large volumes of candidate images using rules; and MagicMix interpolates semantics during the diffusion process. These methods require retraining or heavy computation for each generation (e.g., ConceptLab takes 120s/image, BASS takes 40s/image), lacking practical utility.

**Key Challenge**: Diffusion models can comprehend concrete adjectives ("blue") but fail to grasp abstract adjectives ("creative"). The root cause is that the embedding of "creative" in text encoders lacks the specific semantics required to guide compositional generation.

**Goal**: To make "creative" a semantically clear adjective just like "blue," directly modifying any pair of concepts to achieve zero-shot compositional creativity.

**Key Insight**: Since the problem stems from the overly abstract semantics of the word "creative", it can be redefined using a data-driven approach.

**Core Idea**: Redefine "creative" as a learnable token `<CreTok>` and iteratively optimize its embedding over a large volume of text pairs, enabling it to encode the meta-ability of "how to combine two concepts".

## Method

### Overall Architecture

Based on Stable Diffusion 3, all diffusion model parameters are frozen, and only the embedding vector of a new token `<CreTok>` is optimized in the embedding space of the CLIP text encoder. During training, text pairs are sampled from the CangJie dataset to construct restrictive and adaptive prompts, and the cosine distance between their embeddings is minimized. During inference, "creative" in the prompt is replaced with `<CreTok>` to directly generate images.

### Key Designs

1. **Concept fusion for a single text pair**:

    - **Function**: Achieve token-level semantic fusion of two concepts.
    - **Mechanism**: Given a text pair $(t_1, t_2)$ (e.g., Lettuce, Mantis), construct the restrictive prompt $\mathcal{P}_r$ = "a lettuce mantis" and the adaptive prompt $\mathcal{P}_a$ = "a photo of a `<CreTok>` mixture". The optimization objective is to maximize the cosine similarity between the CLIP text embeddings of the two prompts. A threshold $\theta=0.5$ is introduced to truncate the loss to prevent overfitting. Meanwhile, the loss is calculated for both sequences $(t_1, t_2)$ and $(t_2, t_1)$ to avoid order bias.
    - **Design Motivation**: Operating directly in the text embedding space rather than during the diffusion process keeps the computational cost extremely low and does not alter the generative model parameters. The threshold $\theta$ balances the degree of concept fusion and the risk of overfitting.

2. **Continuous iterative refinement of `<CreTok>`**:

    - **Function**: Enable `<CreTok>` to learn the general meta-ability of "how to make creative combinations," rather than a specific concept.
    - **Mechanism**: Iteratively train on the CangJie dataset (200 training text pairs). In each step, $n=16$ text pairs are randomly sampled to compute the cumulative loss $\mathcal{L}_{iter} = \frac{1}{n}\sum_{i=1}^{n}\tilde{\mathcal{L}}_{mix}^i$, updating the `<CreTok>` embedding. Sampling different text pairs at each step ensures generalization.
    - **Design Motivation**: If trained on only one text pair, the token would encode the semantics of a specific concept (e.g., a specific "lettuce-mantis" hybrid). By iterating across a wide variety of different pairs, the token shifts from "learning a specific mixture" to "learning the meta-ability of how to mix."

3. **CangJie Dataset**:

    - **Function**: Provide diverse text pairs as training materials.
    - **Mechanism**: Combine concept pairs from categories such as animals and plants, consisting of 200 training pairs and 27 test pairs sourced from BASS.
    - **Design Motivation**: Dataset diversity guarantees the generalization ability of `<CreTok>`, enabling it to handle unseen concept pairs after training.

### Loss & Training

- Loss function: Cosine similarity loss with a threshold $\tilde{\mathcal{L}}_{mix} = 1 - \min[\cos(E(\mathcal{P}_r), E(\mathcal{P}_a)), \theta]$
- Bidirectional sequence training to avoid position bias of text pairs.
- Training configuration: 10K steps, single RTX 4090 GPU, LR=0.01 + cosine scheduler, batch size = 1 with gradient accumulation of 16 steps, completed in about 30 minutes.
- Only optimize the embedding vector of `<CreTok>`, leaving all parameters of the diffusion model and text encoders untouched.
- Image-free training process, optimized purely in the text embedding space.

## Key Experimental Results

### Main Results

| Method | VQAScore↑ | PickScore↑ | ImageReward↑ | Generation Speed |
|------|-----------|------------|--------------|---------|
| CreTok | **0.835** | **21.775** | **1.065** | 4s/img |
| SD 3.5 | 0.805 | 21.766 | 0.881 | - |
| Kandinsky 3 | 0.771 | 21.637 | 0.634 | - |
| BASS | 0.710 | 20.799 | 0.481 | 40s/img |
| ConceptLab | - | - | - | 120s/img |

| GPT-4o Score | Integration | Alignment | Originality | Aesthetics | Overall |
|------------|--------|--------|--------|------|------|
| CreTok | **9.5** | **9.9** | **9.3** | **9.6** | **9.6** |
| SD 3.5 | 9.1 | 9.9 | 9.1 | 9.4 | 9.4 |
| BASS | 8.9 | 9.3 | 8.7 | 8.3 | 8.8 |

### Ablation Study

| Configuration | Effect Description |
|------|---------|
| $\theta=0.3$ (Low threshold) | Two concepts are generated independently without fusion. |
| $\theta=0.5$ (Ours) | Optimal balance, concepts fused without overfitting. |
| $\theta=0.7$ (High threshold) | Overfitting to one of the concepts. |
| Training for 2K steps | `<CreTok>` mainly absorbs the semantics of a single concept. |
| Training for 10K steps | `<CreTok>` learns generalized creative meta-abilities. |

### Key Findings

- `<CreTok>` performs well on unseen concept pairs (such as the Lettuce-Mantis combination which never appeared during training), demonstrating the generalization ability of meta-creativity.
- Outperforms larger and stronger models like SD 3.5 and DALL-E 3 on human preference metrics (PickScore, ImageReward).
- Can be seamlessly extended to 3+ concept fusion and creative generation without reference concepts (CT2I task).
- Can be freely paired with style prompts (e.g., "oil painting", "watercolor"), which ConceptLab and BASS cannot support.
- Achieves an average rank of 1.9 in user studies, significantly outperforming other methods (3.1-3.4).

## Highlights & Insights

- **The idea of redefining abstract adjectives as learnable tokens is highly generalizable**: It can be applied not only to "creative", but in theory, any semantically vague adjectives (such as "beautiful" or "scary") can be processed similarly to enhance the model's understanding. This represents a brand-new paradigm for capability enhancement in models.
- **Pure text space optimization without touching diffusion model parameters**: Best-in-class creative generation capability is obtained with only 30 minutes of training and 4 seconds of inference, offering extreme engineering practicality.
- **The distinction between meta-creativity and static creativity is highly inspiring**: While prior methods train separate tokens for each creative output, this work learns the general capability of "how to be creative".
- **Essential difference from personalization methods**: Methods like Textual Inversion represent "what", whereas `<CreTok>` represents "how".

## Limitations & Future Work

- Relies on the text encoding capability of CLIP; if two concepts are too far apart in the CLIP space, the fusion effect might be limited.
- Only validated on compositional creativity (TP2O) tasks, leaving more open-ended creative scenarios (e.g., style innovation, layout innovation) unexplored.
- The CangJie dataset mainly consists of animal and plant categories, with limited concept diversity.
- Only validated on SD 3, without testing generalization on other base models.
- The choice of threshold $\theta=0.5$ is empirical; other base models may require tuning.
- Future work could explore multiple variants of `<CreTok>` and continuous representations of controllable creativity levels.

## Related Work & Insights

- **vs ConceptLab**: Trains an independent token for each new concept (120s each time), whereas this work trains once and applies to all concept pairs in a zero-shot manner (4s).
- **vs BASS**: Relies on rule search plus heavy candidate filtering to obtain creative images (40s), whereas this work directly generates them without relying on search.
- **vs MagicMix/DiffMorpher**: Interpolates during the diffusion process, which heavily relies on reference images and lacks flexibility, whereas this work requires no reference images.
- **vs Textual Inversion**: TI encodes specific visual concepts into tokens, whereas this work encodes abstract semantic capability into tokens, serving different objectives.
- **Insight**: Tokenization of abstract concepts may serve as a general strategy for enhancing foundation model capabilities.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The idea of precisely encoding the abstract concept of "creativity" into a trainable token is highly novel, and the concept of meta-creativity is inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprises multi-dimensional evaluations (automated metrics + GPT-4o + user studies) and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Good storytelling (utilizing the "blue banana" analogy) with clear concepts.
- Value: ⭐⭐⭐⭐ The methodology is highly generalizable, with exceptionally low engineering deployment costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Enhancing Creative Generation on Stable Diffusion-based Models](enhancing_creative_generation_on_stable_diffusion-based_models.md)
- [\[CVPR 2026\] Breaking Semantic Boundaries: Distribution-Guided Semantic Exploration for Creative Generation](../../CVPR2026/image_generation/breaking_semantic_boundaries_distribution-guided_semantic_exploration_for_creati.md)
- [\[CVPR 2025\] DNF: Unconditional 4D Generation with Dictionary-Based Neural Fields](dnf_unconditional_4d_generation_with_dictionary-based_neural_fields.md)
- [\[ICCV 2025\] CAP: Evaluation of Persuasive and Creative Image Generation](../../ICCV2025/image_generation/cap_evaluation_of_persuasive_and_creative_image_generation.md)
- [\[ICLR 2026\] VLM-Guided Adaptive Negative Prompting for Creative Generation](../../ICLR2026/image_generation/vlm-guided_adaptive_negative_prompting_for_creative_generation.md)

</div>

<!-- RELATED:END -->
