---
title: >-
  [Paper Note] Personalized Preference Fine-tuning of Diffusion Models
description: >-
  [CVPR 2025][Image Generation][Personalized Preference Alignment] PPD proposes a personalized preference diffusion model fine-tuning framework: it leverages a VLM to extract user embeddings from a small number of (4 pairs) preference examples, injects them into the diffusion model via decoupled cross-attention layers, and optimizes multi-user personalized preferences simultaneously using a DPO objective. It requires only 4 preference pairs to generate preference-aligned images…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Personalized Preference Alignment"
  - "DPO"
  - "Multi-Reward Optimization"
  - "VLM User Embedding"
  - "Diffusion Model Fine-Tuning"
date: 2026-05-08
content_hash: f2d3143cb55ed614
---

# Personalized Preference Fine-tuning of Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2501.06655](https://arxiv.org/abs/2501.06655)  
**Code**: None  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Personalized Preference Alignment, DPO, Multi-Reward Optimization, VLM User Embedding, Diffusion Model Fine-Tuning

## TL;DR
PPD proposes a personalized preference diffusion model fine-tuning framework: it leverages a VLM to extract user embeddings from a small number of (4 pairs) preference examples, injects them into the diffusion model via decoupled cross-attention layers, and optimizes multi-user personalized preferences simultaneously using a DPO objective. It requires only 4 preference pairs to generate preference-aligned images for a new user (76% win rate).

## Background & Motivation

**Background**: RLHF techniques such as Diffusion-DPO have significantly improved the generation quality of text-to-image diffusion models. However, these methods optimize a single reward function representing group-level average preferences, neglecting the unique tastes of individual users—for instance, some prefer bright colors, while others prefer centered foregrounds.

**Limitations of Prior Work**: (1) Fine-tuning a separate model for each user is not scalable; (2) Methods like IP-Adapter control generation via reference images, but they are limited to a single image input and do not directly learn preferences; (3) Although preference datasets (such as Pick-a-Pic) contain user ID annotations, user characteristic information is extremely scarce, as user self-reported preferences are often inaccurate.

**Key Challenge**: Personalization requires understanding "what users prefer," but such preferences are difficult to describe precisely with text or a single image, and are better expressed implicitly through pairwise comparisons ("A is better than B"). However, how to extract user representations from a small number of pairwise comparisons for conditional generation models remains an unresolved problem.

**Goal**: Design a unified framework that allows a single diffusion model to simultaneously learn the personalized preferences of multiple users and generalize to unseen users at training time.

**Key Insight**: Formulate the personalization problem as a conditional generation task, where user preferences are injected into the diffusion model as extra conditions. Key insight: The intermediate hidden states of a VLM (such as LLaVA-OneVision) can effectively encode user characteristics from a few preference pairs.

**Core Idea**: Process 4 sets of preference examples (each containing a text prompt, preferred image, and non-preferred image) using a VLM to extract user embeddings. These embeddings are then injected into Stable Cascade via IP-Adapter-like decoupled cross-attention, enabling joint training using a user-conditioned Diffusion-DPO objective.

## Method

### Overall Architecture
PPD consists of two stages: Stage 1 (User Embedding Generation): Sample $N=4$ sets of preference examples for each user from the preference dataset and feed them into a VLM to extract intermediate hidden states as user embeddings. Stage 2 (Conditional Fine-Tuning): Insert decoupled cross-attention layers in Stage C of Stable Cascade to process user embeddings, and fine-tune the model with a user-conditioned DPO objective (only training the added cross-attention layers while freezing the pretrained model).

### Key Designs

1. **VLM User Embedding**:

    - **Function**: Extract vectors from a few preference examples that can effectively represent user preferences.
    - **Mechanism**: Use LLaVA-OneVision to process 4 sets of preference examples per user (text + preferred image + non-preferred image) to extract intermediate hidden states as user embeddings. Validation experiments show that a user classifier trained on frozen embeddings achieves a Top-16 accuracy of 90% among 300 users, demonstrating that the embeddings can effectively distinguish different users' preferences.
    - **Design Motivation**: One-hot encoding cannot generalize to new users. VLM embeddings have semantic meaning (based on the visual content of the preference examples) and naturally support zero-shot generalization—new users only need to provide 4 preference pairs to generate embeddings. Under the Bradley-Terry model, characteristics of preference pairs serve as sufficient statistics for the reward function.

2. **Personalized DPO Objective**:

    - **Function**: Enable a single model to optimize personalized preferences of multiple users simultaneously, rather than a group-level average preference.
    - **Mechanism**: Extend the standard Diffusion-DPO to a user-conditioned version $L_{PPD}(\theta) = -\mathbb{E}_{c, x_0^+, x_0^-, u}[\log\sigma(-\beta T\omega(\lambda_t)\Delta)]$, where the noise predictor $\epsilon_\theta(x_t, c, u, t)$ is additionally conditioned on the user embedding $u$. Each training sample indicates which user the preference came from. The model learns to generate images in different styles based on the corresponding user embeddings. During training, user embeddings are randomly dropped out (zeroed out) for regularization.
    - **Design Motivation**: Standard DPO optimizes a single preference direction for all users, which suppresses minority preferences. User conditioning allows the model to maintain independent preference directions for each user.

3. **Decoupled Cross-Attention Conditioning**:

    - **Function**: Inject user embeddings into the diffusion model while keeping the text conditioning unaffected.
    - **Mechanism**: Following the design of IP-Adapter, a new cross-attention layer is added alongside each text cross-attention layer to process user embeddings: $Z' = \text{Softmax}(\frac{QK^T}{\sqrt{d}})V + \text{Softmax}(\frac{Q(K')^T}{\sqrt{d}})V'$, where $K' = u_t W_k'$ and $V' = u_t W_v'$. Only the added parameters $W_k'$ and $W_v'$ are trained, while the pretrained model is frozen.
    - **Design Motivation**: The decoupled design ensures that user preferences and text semantics are independent conditioning channels. User preferences influence high-level attributes such as style, color, and composition, while the text controls semantic content. Training only the newly added layers ensures training efficiency and maintains the quality of the base model.

### Loss & Training
Using the Pick-a-Pic v2 dataset (58K text prompts, 0.8M image pairs, 5K users), the newly added cross-attention layers are fine-tuned using the user-conditioned Diffusion-DPO target. Optimized with AdamW, effective batch size of 768 pairs, learning rate of $1\times10^{-5}$, and trained for 1 epoch.

## Key Experimental Results

### Main Results
Multi-reward optimization (synthetic user experiment, where each reward function acts as a "user"):

| Method | CLIP↑ | Aesthetic↑ | HPS↑ |
|------|-------|-----------|------|
| Stable Cascade | 31.97 | 5.33 | 23.87 |
| Diffusion-DPO | 32.48 | 5.46 | 25.96 |
| SFT | 32.26 | 5.56 | 25.78 |
| **PPD (ours)** | **32.66** | **5.92** | **27.51** |
| DPO (CLIP only) | 32.96 | - | - |
| DPO (Aesthetic only) | - | 6.42 | - |
| DPO (HPS only) | - | - | 28.61 |

### Ablation Study

| Configuration | Description |
|------|------|
| One-hot user encoding | Cannot generalize to new users |
| VLM user embedding | Generalizes, Top-16 classification accuracy of 90% |
| w/o user dropout | Overfits training users |
| w/ user dropout | Better generalization |
| Pick-a-Pic real users | 76% win rate vs. Stable Cascade |

### Key Findings
- PPD simultaneously optimizes three reward functions and approaches their respective individual upper bounds, proving that a single model can effectively accommodate multiple preferences.
- Different preferences can be smoothly transitioned during inference by linearly interpolating reward weights (Figure 4).
- In Pick-a-Pic real-user scenarios, transitioning with just 4 preference pairs achieves a 76% win rate against Stable Cascade.
- The classification accuracy of VLM embeddings among 300 users is far superior to random baselines, proving its ability to distinguish preferences.

## Highlights & Insights
- **Preference-as-Condition** Paradigm Shift: Transforming personalized preference from an "optimization target" into a "generation condition," making it possible for a single model to serve multiple users. During inference, one only needs to switch user embeddings without retraining.
- **VLM as Preference Encoder**: Utilizing the multi-image understanding capability of VLMs to extract user characteristics from preference pairs. This idea can be migrated to LLM personalization, recommendation systems, and other fields.
- **Reward Interpolation**: Smooth interpolation between different preferences can be performed during inference, providing unprecedented flexibility in generation control.

## Limitations & Future Work
- Currently validated only on Stable Cascade; newer architectures like SDXL and Flux have not been tested.
- Four preference pairs may not be sufficient to describe complex preferences.
- The interpretability of the user embeddings is limited, making it difficult to understand exactly "what preferences" the model has learned.
- The risk of negative transfer during multi-user simultaneous training has not been thoroughly analyzed.

## Related Work & Insights
- **vs Diffusion-DPO**: DPO optimizes group-level average preferences, while PPD focuses on individual preferences. PPD approaches the single-reward upper bound of DPO while simultaneously optimizing multiple rewards.
- **vs IP-Adapter**: IP-Adapter extracts style information from reference images, whereas PPD extracts preference details from preference pairs. The conditioning information of PPD is more abstract (preferences rather than style).
- **vs PRISM/Pluralistic Alignment**: Pluralistic value alignment work in the LLM domain. PPD introduces similar concepts to the image generation domain.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Preference-as-condition" framework design, VLM preference encoder, and personalized DPO integration are highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic and real-user experiments are extensive, but lack a large-scale user study.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, and the method derivation is rigorous.
- Value: ⭐⭐⭐⭐⭐ Opens up a new direction for diffusion model personalization, with a framework that is both general and practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models](efficient_fine-tuning_and_concept_suppression_for_pruned_diffusion_models.md)
- [\[CVPR 2025\] SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-Image Diffusion Models](sleepermark_towards_robust_watermark_against_fine-tuning_text-to-image_diffusion.md)
- [\[CVPR 2025\] Calibrated Multi-Preference Optimization for Aligning Diffusion Models](calibrated_multi-preference_optimization_for_aligning_diffusion_models.md)
- [\[CVPR 2025\] Curriculum Direct Preference Optimization for Diffusion and Consistency Models](curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)
- [\[CVPR 2025\] Reward Fine-Tuning Two-Step Diffusion Models via Learning Differentiable Latent-Space Surrogate Reward](reward_fine-tuning_two-step_diffusion_models_via_learning_differentiable_latent-.md)

</div>

<!-- RELATED:END -->
