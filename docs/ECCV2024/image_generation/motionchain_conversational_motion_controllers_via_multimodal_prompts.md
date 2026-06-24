---
title: >-
  [Paper Note] MotionChain: Conversational Motion Controllers via Multimodal Prompts
description: >-
  [ECCV 2024][Image Generation] This paper proposes MotionChain, a unified vision-motion-language model that generates continuous, long-term human motion sequences across multi-turn conversations via multimodal prompts, supporting the joint understanding and generation of text, images, and motion.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: 71b22568a4076e8c
---

# MotionChain: Conversational Motion Controllers via Multimodal Prompts

**Conference**: ECCV 2024  
**arXiv**: [2404.01700](https://arxiv.org/abs/2404.01700)  
**Area**: Multimodal VLM

## TL;DR

This paper proposes MotionChain, a unified vision-motion-language model that generates continuous, long-term human motion sequences across multi-turn conversations via multimodal prompts, supporting the joint understanding and generation of text, images, and motion.

## Background & Motivation

- **Success of Large Language Models (LLMs)**: LLMs excel in multi-turn conversation and context maintenance, yet this capability remains largely unexplored in the field of human motion generation.
- **Limitations of Prior Work**: Approaches like MDM, MLD, and MotionGPT treat motion generation tasks as single-turn conditional generation, lacking contextual understanding and multi-turn continuous generation capabilities.
- **Application Demands**: Humanoid robots and game agents require the ability to execute human tasks progressively through intuitive multi-turn interactions.
- **Data Scarcity Challenge**: Compared to paired datasets such as image-language and image-pose, text-motion paired data is extremely limited.
- **Core Observation**: Both human motion and natural language are sequential and can be "written" continuously. Consequently, vision-language instruction tuning methodologies can be leveraged to achieve conversational motion generation.

## Method

### Overall Architecture

MotionChain consists of three core components:
1. **Multimodal Tokenizer**: Encodes text, images, and motion into a unified space of discrete tokens.
2. **Vision-Motion-Aware Language Model**: Based on a pre-trained language model (Flan-T5) to comprehend multimodal inputs and generate motion/text responses.
3. **Motion Composition Mechanism**: Achieves continuous transitions between multi-turn motions via token concatenation.

### Key Designs

**Motion Tokenizer (VQ-VAE)**:
- Employs a 1D convolutional encoder to encode the motion sequence $m^{1:M}$ into latent vectors, which are then quantized and mapped to a discrete codebook.
- Codebook size is $K \in \mathbb{R}^{512 \times 1024}$ with a temporal downsampling rate of $l=4$.
- Training loss: reconstruction loss $\mathcal{L}_r$ + embedding loss $\mathcal{L}_e$ + commitment loss $\mathcal{L}_c$.

**Vision Tokenizer**:
- Image input: Uses a frozen CLIP ViT-L/14 with a learnable linear projection to map visual features to the language token embedding space.
- Video input: Utilizes CLIP encoding, temporal embedding, and a Perceiver module to aggregate spatial-temporal features.

**Motion Joint (Tokens-joint)**:
- Concatenates the motion tokens of the previous turn $z_p^{1:L_p}$ with those of the current turn $z_c^{1:L_c}$ and decodes them jointly.
- The decoder is capable of achieving smooth transitions between motions at the token level, which outperforms frame-level concatenation.

**Multi-turn Conversation Datasets Construction**:
- Leverages ChatGPT to generate motion reasoning data (analyzing motion context, preceding/succeeding actions, characters, etc.).
- Uses the TMR (Text-to-Motion Retrieval) model to categorize motions by similarity, generating editing instructions for motion pairs with medium similarity.
- Combines single-turn tasks with subsequent tasks (such as translation, reasoning, editing) to form multi-turn conversational data of up to 10 turns.

### Loss & Training

The training objective of the language model is to maximize the log-likelihood of the target answer tokens:

$$\mathcal{L}_{LM} = -\sum_{i=0}^{L_t-1} \log p_\theta(x_a^i | X_v, X_{s,<i}, X_{a,<i})$$

Three-stage training strategy: (1) motion tokenizer pre-training $\rightarrow$ (2) motion-language pre-training $\rightarrow$ (3) instruction tuning.

## Key Experimental Results

### Main Results

**Motion Reasoning Comparison (vs. Large Language Models)**:

| Method | Parameters | Bleu@1↑ | Bleu@4↑ | Rouge↑ | Cider↑ | BertScore↑ |
|------|--------|---------|---------|--------|--------|------------|
| Flan-T5-base | 250M | 4.64 | 1.78 | 15.32 | 15.93 | 3.45 |
| Llama-2-7b | 7B | 11.12 | 3.67 | 19.14 | 1.04 | 6.81 |
| Vicuna-1.5-7b | 7B | 19.27 | 7.39 | 25.75 | 5.44 | 19.05 |
| Vicuna-1.5-13b | 13B | 17.20 | 6.53 | 24.18 | 7.77 | 18.00 |
| **MotionChain** | **280M** | **37.92** | **19.19** | **38.05** | **24.53** | **32.24** |

**Sequential Motion Composition Comparison (BABEL Dataset)**:

| Method | Diversity | MPJPE↓ | PA-MPJPE↓ | ACCL↓ |
|------|-----------|--------|-----------|-------|
| Real | 15.74 | - | - | - |
| TEACH | 27.11 | 979.21 | 933.32 | 23.02 |
| **MotionChain** | **43.25** | **276.05** | **53.72** | **7.11** |

### Ablation Study

**Comparison of Motion Composition Mechanisms (HumanML3D)**:

| Composition Method | MPJPE↓ | PA-MPJPE↓ | ACCL↓ | Diversity |
|----------|--------|-----------|-------|-----------|
| Independent | 350.79 | 102.97 | 11.40 | 6.47 |
| Past-condition | 232.46 | 46.15 | 6.18 | 6.01 |
| **Tokens-joint** | **108.77** | **18.85** | **2.26** | 5.56 |

**Comparison of Vision Tokenizer Architectures (BEDLAM)**:

| Architecture | First-frame MPJPE↓ | First-frame PA-MPJPE↓ | Last-frame MPJPE↓ | Last-frame PA-MPJPE↓ |
|------|-------------------|-----------------------|-------------------|---------------------|
| Q-former | 195.49 | 86.56 | 134.73 | 57.17 |
| Perceiver | 185.61 | 99.21 | 134.89 | 57.58 |
| **Linear** | **144.37** | **76.48** | **133.73** | **56.73** |

### Key Findings

1. MotionChain with only 280M parameters significantly outperforms 7B-13B LLMs in motion reasoning tasks, validating the necessity of motion awareness.
2. Compared to independent decoding, the Tokens-joint composition method reduces MPJPE from 350.79 to 108.77 (a 69% reduction), demonstrating the effectiveness of token-level motion composition.
3. A simple linear projection in the vision tokenizer is sufficient for understanding human poses, rendering more complex architectures like Q-former or Perceiver unnecessary.
4. In sequential motion composition, MotionChain achieves a PA-MPJPE that is only 5.8% of TEACH's (53.72 vs. 933.32), presenting a substantial improvement in motion quality.

## Highlights & Insights

- **Unified Representation**: Encodes motion, text, and images uniformly as tokens, enabling multimodal understanding and generation via a unified language model architecture.
- **Conversational Control**: Establishes multi-turn conversation-driven continuous motion generation for the first time, where each turn's output is conditioned on all prior dialogue context.
- **Token-Level Motion Joint**: By concatenating motion tokens and decoding them jointly, the model achieves smoother and more natural transitions compared to frame-level concatenation.
- **Small Model with Great Capability**: The 280M-parameter model outperforms 13B LLMs, suggesting that domain-specific motion perception is more critical than merely scaling up the model size.

## Limitations & Future Work

- Based on a probabilistic generative model, it cannot provide the deterministic control characteristics of traditional motion controllers.
- It is limited to generating articulated body motions and does not include fine-grained motions such as hand gestures or facial expressions.
- It lacks collision signals, making it unable to handle human-object and human-scene interactions.

## Rating

⭐⭐⭐⭐ (4/5) — This work introduces multi-turn conversations into the motion generation field for the first time with an elegantly designed unified multimodal representation. However, it is constrained by the scale of motion data and interaction modeling capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[ECCV 2024\] SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)

</div>

<!-- RELATED:END -->
