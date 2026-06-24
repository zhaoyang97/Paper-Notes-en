---
title: >-
  [Paper Note] Enhancing Diffusion Models with Text-Encoder Reinforcement Learning
description: >-
  [ECCV 2024][Image Generation][Diffusion Models] This paper proposes TexForce, which utilizes reinforcement learning (DDPO) combined with LoRA to fine-tune the text encoder of diffusion models, thereby improving text-image alignment and visual quality. It can be seamlessly combined with existing U-Net fine-tuning methods to achieve superior performance.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Diffusion Models"
  - "Text Encoder"
  - "Reinforcement Learning"
  - "LoRA"
  - "Text-Image Alignment"
date: 2026-05-08
content_hash: 5a61d688d5ce9ff4
---

# Enhancing Diffusion Models with Text-Encoder Reinforcement Learning

**Conference**: ECCV 2024  
**arXiv**: [2311.15657](https://arxiv.org/abs/2311.15657)  
**Code**: [GitHub](https://github.com/chaofengc/TexForce)  
**Area**: Image Generation  
**Keywords**: Diffusion Models, Text Encoder, Reinforcement Learning, LoRA, Text-Image Alignment

## TL;DR

This paper proposes TexForce, which utilizes reinforcement learning (DDPO) combined with LoRA to fine-tune the text encoder of diffusion models, thereby improving text-image alignment and visual quality. It can be seamlessly combined with existing U-Net fine-tuning methods to achieve superior performance.

## Background & Motivation

Current text-to-image diffusion models (such as Stable Diffusion) are trained to optimize log-likelihood, which creates a gap between this objective and the specific requirements of downstream tasks (e.g., image aesthetics, text-image alignment). Existing improvement methods mainly fine-tune the U-Net through reinforcement learning (DDPO, DPOK) or direct backpropagation (ReFL, AlignProp). However, several key issues remain:

**Neglecting the Text Encoder**: Almost all methods keep the pretrained text encoder frozen and only tune the U-Net. However, the text encoder itself is sub-optimal—users often need to carefully engineer prompts to obtain satisfactory results.

**Side Effects of U-Net Fine-tuning**: Fine-tuning the U-Net can easily disrupt the visual appearance of images, leading to style degradation or mode collapse.

**Text Encoder as a Bottleneck**: Even if the U-Net is fine-tuned, the sub-optimal text encoder still limits overall performance.

The authors observe a key phenomenon: **Fine-tuning the U-Net tends to improve reward scores by changing the visual appearance, whereas fine-tuning the text encoder achieves the same goal by introducing new visual concepts, with the latter being superior in maintaining semantics.**

## Method

### Overall Architecture

The core of TexForce is straightforward: using the DDPO (Denoising Diffusion Policy Optimization) algorithm, LoRA fine-tuning is applied to the text encoder $\tau_\phi$, which is optimized based on task-specific reward functions. The key features are:

1. The text encoder is fine-tuned via RL, while the U-Net is frozen.
2. Parameter-efficient fine-tuning is achieved using LoRA.
3. Simple combination with existing fine-tuning methods of U-Net is enabled without extra training.

### Key Designs

#### Reinforcement Learning in Diffusion Models

The denoising process is formalized as a Markov Decision Process. The text encoder $\tau_\phi$ acts as a policy network that maps the text description to actions (text embeddings), thereby influencing the generation process of the diffusion model. The optimization goal is to maximize the expected reward:

$$J(\phi) = \mathbb{E}[R(\mathbf{x}_0, s)]$$

The policy gradient can be computed as:

$$\nabla_\phi J = \mathbb{E}\left[\sum_{t=0}^{T} \nabla_\phi \log p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t, \tau_\phi(s)) R(\mathbf{x}_0, s)\right]$$

In practice, PPO (Proximal Policy Optimization) is used to stabilize training:

$$J = \mathbb{E}[\min(r_t(\phi)A, \text{clip}(r_t(\phi), 1-\lambda, 1+\lambda)A)]$$

where $A$ is the advantage value of the normalized reward, and $r_t$ is the probability ratio of the new policy to the old policy.

#### LoRA Low-Rank Adaptation

Trainable low-rank matrices are inserted into the feed-forward layers of the text encoder: $W' = W + \alpha \Delta W$, where $\Delta W$ is initialized to zero. The advantages of LoRA are:

1. **Overfitting Prevention**: The low-rank constraint limits the parameter space.
2. **Flexible Switching**: LoRA weights for different tasks can be flexibly replaced.
3. **Weight Merging**: LoRA weights for different tasks can be directly weight-blended to combine capabilities.

#### Why Fine-tune the Text Encoder

From a theoretical perspective, when optimizing the ELBO with a small number of prompts $s$ during the fine-tuning stage:

$$\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}|s)}[\log(p_\theta(\mathbf{x}_{0:T}|\mathbf{z}))] - D_{KL}(q_\phi(\mathbf{z}|s) \| p(\mathbf{z}))$$

When $\phi$ is fixed, $q_\phi$ may serve as a sub-optimal estimation of $p(\mathbf{z})$, leading to an increase in the KL divergence term. Therefore, **fine-tuning the text encoder $\tau_\phi$ can reduce the KL term**, which is particularly important under limited data.

#### Direct Fusion of U-Net and Text Encoder LoRA Weights

The most important design is that the text encoder LoRA weights in TexForce can be **directly combined with existing U-Net LoRA weights** without any additional training. Experiments demonstrate that this simple fusion strategy consistently outperforms joint training.

The authors hypothesize that physical reason: **The fixed U-Net acts as a pixel generation prior during text encoder fine-tuning**, and joint fine-tuning would make the optimization of the text encoder more complex.

### Loss & Training

- **Reward Functions**: Supports various non-differentiable rewards, including ImageReward, HPSv2, JPEG compression size, face quality score, and hand detection confidence.
- **Optimization Algorithm**: PPO with importance sampling.
- **LoRA Fusion**: $\sum_i \alpha_i \theta_i$, where LoRA weights for different tasks are combined via weighted summation.

## Key Experimental Results

### Main Results

**Quantitative Results on ImageReward Dataset**

| Method | ImageReward Score↑ |
|------|-----------------|
| SDv1.4 | 0.2154 |
| ReFL (U-Net fine-tuning) | 0.4485 |
| TexForce (Text Encoder fine-tuning) | 0.4556 |
| ReFL + TexForce | **0.6553** |

**Quantitative Results on HPSv2 Dataset**

| Method | HPSv2 Score↑ |
|------|-----------|
| SDv1.4 | 0.2752 |
| AlignProp (U-Net fine-tuning) | 0.2821 |
| TexForce | 0.2767 |
| AlignProp + TexForce | **0.2914** |

**Multi-backbone Results (ImageReward)**

| Backbone | Original | ReFL | TexForce | ReFL+TexForce |
|----------|------|------|----------|--------------|
| SDv1.5 | 0.2140 | 0.5484 | 0.4086 | **0.6703** |
| SDv2.1 | 0.3891 | 0.5223 | 0.5084 | **0.6158** |

### Ablation Study

**Simple Fusion vs. Joint Training**

| Method | ImageReward Score |
|------|----------------|
| SDv1.4 | 0.2154 |
| ReFL Only | 0.4485 |
| TexForce Only | 0.4556 |
| Joint Training | 0.5009 |
| Simple Fusion (ReFL+TexForce) | **0.6553** |

Simple fusion significantly outperforms joint training, validating the importance of keeping U-Net fixed as a prior.

### Key Findings

1. **Behavioral Difference**: Fine-tuning the U-Net alters visual appearance to pursue rewards, whereas fine-tuning the text encoder introduces new concepts to achieve the same goal—the latter preserves semantics better.
2. **Complementarity**: The capacities learned by both are complementary; a simple combination of them outperforms joint training.
3. **GPT-4V Evaluation**: TexForce scores the highest in text-image alignment, ReFL performs better in visual appearance, and the combined scheme achieves the best overall performance.
4. **Cross-backbone Robustness**: The method is effective across SDv1.4, SDv1.5, and SDv2.1, consistently improving even the already strong SDv2.1.
5. **Mixable LoRA Weights**: LoRA weights trained for different tasks (e.g., ImageReward + face quality) can be directly blended to enhance specific target qualities.

## Highlights & Insights

1. **Novel Perspective**: While almost all concurrent works focus on U-Net fine-tuning, this paper is among the earliest to systematically study text encoder fine-tuning.
2. **Simple yet Powerful Combination Strategy**: Merges the advantages of different fine-tuning schemes without requiring additional training, significantly lowering the barrier to practical deployment.
3. **Theoretical and Empirical Consistency**: Provides theoretical motivation for fine-tuning the text encoder starting from ELBO analysis, which is perfectly validated by experiments.
4. **High Flexibility**: Does not require differentiable reward functions; any evaluation metric for image quality can serve as a reward.

## Limitations & Future Work

1. RL training is slower and more computationally heavy than direct backpropagation (as shown in Figure 4, optimizing the text encoder is more challenging).
2. Validated only on the Stable Diffusion series; SDXL or newer architectures are not tested.
3. The fusion coefficient $\alpha_i$ for combining different reward-trained LoRA weights requires manual tuning.
4. Generalization capability of the fine-tuned text encoder to out-of-distribution prompts has not been studied.
5. Lacks direct comparison with the concurrent work TextCraftor.

## Related Work & Insights

- **DDPO**: The base RL algorithm used in this work, formulating the diffusion denoising process as an MDP.
- **ReFL / AlignProp / DRaFT**: Direct backpropagation methods, which are more prone to overfitting and mode collapse.
- **TextCraftor**: A concurrent work exploring text encoder fine-tuning, but it relies on differentiable rewards and is thus less flexible.
- Insight: The concept of text encoder fine-tuning can be extended to diffusion models in other modalities, such as video generation and 3D generation.

## Rating

| Dimension | Rating (1-5) |
|------|-----------|
| Novelty | 4 |
| Theoretical Depth | 3.5 |
| Experimental Thoroughness | 4.5 |
| Practical Value | 4.5 |
| Writing Quality | 4 |
| Overall Rating| 4.1 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Powerful and Flexible: Personalized Text-to-Image Generation via Reinforcement Learning](powerful_and_flexible_personalized_texttoimage_generation_vi.md)
- [\[ECCV 2024\] LCM-Lookahead for Encoder-Based Text-to-Image Personalization](lcmlookahead_for_encoderbased_texttoimage_personalization.md)
- [\[ICLR 2026\] Consolidating Reinforcement Learning for Multimodal Discrete Diffusion Models](../../ICLR2026/image_generation/consolidating_reinforcement_learning_for_multimodal_discrete_diffusion_models.md)
- [\[ICML 2026\] ForceForget: Reinforcement Concept Removal for Enhancing Safety in Text-to-Image Models](../../ICML2026/image_generation/forceforget_reinforcement_concept_removal_for_enhancing_safety_in_text-to-image_.md)
- [\[ECCV 2024\] Lego: Learning to Disentangle and Invert Personalized Concepts Beyond Object Appearance in Text-to-Image Diffusion Models](lego_learning_to_disentangle_and_invert_personalized_concepts_beyond_object_appe.md)

</div>

<!-- RELATED:END -->
