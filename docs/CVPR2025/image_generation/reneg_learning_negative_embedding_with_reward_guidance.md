---
title: >-
  [Paper Note] ReNeg: Learning Negative Embedding with Reward Guidance
description: >-
  [CVPR 2025][Image Generation][Negative Prompt Optimization] ReNeg proposes directly learning negative embeddings in a continuous text embedding space guided by a reward model, replacing handcrafted negative prompts. By optimizing a minimal number of parameters, it matches the generation quality of full-model fine-tuning methods on the HPSv2 benchmark. Furthermore, the learned embeddings can be directly transferred to other T2I and T2V models.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Negative Prompt Optimization"
  - "Reward Guidance"
  - "Diffusion Models"
  - "Classifier-Free Guidance"
  - "Human Preference Alignment"
date: 2026-05-08
content_hash: 5c09ce7261306f79
---

# ReNeg: Learning Negative Embedding with Reward Guidance

**Conference**: CVPR 2025  
**arXiv**: [2412.19637](https://arxiv.org/abs/2412.19637)  
**Code**: Yes (see paper URL)  
**Area**: Image Generation  
**Keywords**: Negative Prompt Optimization, Reward Guidance, Diffusion Models, Classifier-Free Guidance, Human Preference Alignment

## TL;DR
ReNeg proposes directly learning negative embeddings in a continuous text embedding space guided by a reward model, replacing handcrafted negative prompts. By optimizing a minimal number of parameters, it matches the generation quality of full-model fine-tuning methods on the HPSv2 benchmark. Furthermore, the learned embeddings can be directly transferred to other T2I and T2V models.

## Background & Motivation

**Background**: Classifier-Free Guidance (CFG) is a core technology for improving generation quality in diffusion models. In practice, replacing the null-text in CFG with handcrafted negative prompts (e.g., "low resolution, distorted") further enhances performance and is widely adopted in the community.

**Limitations of Prior Work**: Handcrafted negative prompts have fundamental limitations: (1) Incomplete search space—humans cannot exhaustively list all effective combinations of negative words; (2) Inefficient, relying on subjective judgment and trial-and-error; (3) The discrete language space is unsuitable for gradient-based optimization. DNP and DPO-Diff attempt automated search but are still limited by discrete spaces or indirect methods.

**Key Challenge**: Negative embeddings significantly affect generation quality, yet current methods either search inefficiently in a limited discrete space or directly fine-tune the entire model, which is parameter-heavy and may disrupt pretrained knowledge.

**Goal**: Direct learning of optimal negative embeddings in a continuous text embedding space under the guidance of a reward model, achieving generation quality improvement with minimal parameter overhead.

**Key Insight**: Through Jacobian matrix analysis, the authors find that the parameter efficiency of negative embedding, $E(n) = 5.1 \times 10^{-4}$, is significantly higher than that of full-model parameters, $E(\theta_0) = 1.5 \times 10^{-6}$, and LoRA parameters. This indicates that tuning negative embeddings can induce the largest distributional change with minimal parameter updates.

**Core Idea**: Integrates CFG into the training process (instead of using it only during inference), directly optimizing negative embeddings via gradient descent in the continuous embedding space guided by reward model feedback.

## Method

### Overall Architecture
The training pipeline is built upon the pretrained SD1.5 and the HPSv2.1 reward model. During training, the negative embedding is registered as a learnable parameter (initialized from the null-text embedding) while all other model parameters are frozen. Prompts are randomly sampled to generate intermediate denoising results through diffusion sampling with CFG. A one-step prediction is applied to obtain the predicted image $\hat{x}_0$, which is then fed into the reward model to compute scores. Finally, gradients are backpropagated to update the negative embedding. The approach features two learning strategies: global and sample-wise.

### Key Designs

1. **Integrating CFG into Training for Gradient Propagation**:

    - **Function**: Enables the negative embedding to receive gradient signals from the reward model.
    - **Mechanism**: Traditional CFG is applied only during inference as $\tilde{\epsilon}_\theta(x_t,c,t) = \epsilon_\theta(x_t,\phi,t) + \gamma(\epsilon_\theta(x_t,c,t) - \epsilon_\theta(x_t,\phi,t))$, where $\phi$ is the null-text. ReNeg also performs CFG during training, replacing $\phi$ with the learnable $n$. It then obtains the predicted image via one-step prediction $\hat{x}_0 = (x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,c,t))/\sqrt{\bar{\alpha}_t}$, calculates the reward, and backpropagates gradients $\partial\mathcal{J}/\partial n$.
    - **Design Motivation**: The negative embedding only influences generation results within CFG. If CFG is not used during training, gradients cannot propagate to the negative embedding. Integrating CFG into the training process is a crucial prerequisite to make this approach viable.

2. **Deterministic ODE Sampler for Optimizing $\hat{x}_0$ Prediction**:

    - **Function**: Improves the accuracy of the one-step prediction at intermediate timesteps.
    - **Mechanism**: The DDIM (deterministic ODE) sampler is utilized instead of DDPM (stochastic SDE) from $x_T$ to $x_t$, making the prediction of $\hat{x}_0$ closer to the fully-sampled $x_0$. With $T=30$, $t$ is randomly sampled from $[0, 10]$, a range in which reward scores for generations of varying quality are well-distinguished.
    - **Design Motivation**: Reward guidance operates on $\hat{x}_0$; if $\hat{x}_0$ deviates significantly, the reward signals become inaccurate, leading to unreliable learned embeddings. DDIM eliminates stochasticity, stabilizing the gradient signals.

3. **Global + Sample-wise Bi-level Optimization Strategy**:

    - **Function**: Learns a universal negative embedding first, and then adapts it to specific prompts.
    - **Mechanism**: In the global phase, a universal negative embedding applicable to all prompts is learned by training for 4000 steps on 10K prompts. In the sample-wise phase, initialized with the global embedding, additional optimization of up to 10 steps (including early stopping with patience=3) is performed for each specific prompt, ensuring performance is not inferior to the global embedding.
    - **Design Motivation**: Since optimal negative embeddings vary across different prompts (e.g., realistic photos vs. cartoons), sample-wise optimization allows further adaptation, converging in very few steps when initialized from the global embedding.

### Loss & Training
The optimization objective is to maximize the expected reward, formulated as:

$$\mathcal{J}_\theta(\mathcal{D}) = \mathbb{E}_{c \sim \mathcal{D}}(\mathcal{R}(c, \hat{x}_0))$$

Using the AdamW optimizer and a Cosine Scheduler, the model is trained for 4000 steps with an LR of $5 \times 10^{-3}$ and a batch size of 64. Gradients propagate between $\hat{x}_0$ and $x_t$, and are not backpropagated to earlier timesteps.

## Key Experimental Results

### Main Results

| Method | HPSv2 Avg | Parti PickScore↑ | Parti Aesthetic↑ | Parti HPSv2.1↑ |
|------|-----------|-------------------|------------------|----------------|
| SD1.5 | 25.21 | 18.40 | 5.23 | 25.67 |
| SD1.5 + Handcrafted Negative Prompt | 26.76 | 19.14 | 5.26 | 26.79 |
| Diffusion-DPO (Fine-tuned) | 26.68 | 19.48 | 5.26 | 26.62 |
| ReFL (Fine-tune UNet) | 28.27 | 18.17 | 5.48 | 27.97 |
| TextCraftor (Fine-tuning) | 29.87 | 19.17 | 5.90 | 28.36 |
| **ReNeg Global** | **31.08** | **19.90** | 5.45 | **29.16** |
| **ReNeg Sample-wise** | **31.89** | **19.97** | **5.50** | **29.84** |

### Ablation & Transfer Studies

| Model Transfer | Aesthetic Quality↑ | Motion Smoothness↑ | Background Consistency↑ |
|---------|----------|----------|------------|
| VideoCrafter2 | 58.0 | 97.7 | 97.6 |
| + Handcrafted Negative Prompt | 57.8 | 97.8 | 97.9 |
| + **ReNeg** | **58.6** | **97.8** | **98.5** |
| ZeroScope | 49.9 | 98.9 | 97.7 |
| + **ReNeg** | **58.7** | 98.1 | **98.7** |

### Key Findings
- ReNeg Global embedding completely outperforms handcrafted negative prompts across all four categories of style in HPSv2, and even surpasses TextCraftor (31.08 vs 29.87) which requires full-model fine-tuning.
- Substantial cross-model transfer performance: The embedding learned on SD1.5 can be directly transferred to the ZeroScope video model, improving the aesthetic quality from 49.9 to 58.7 (+17.6%).
- The win rate on SD1.4/1.5/2.1 consistently exceeds 90%, demonstrating strong generalization across different SD versions.
- Parameter efficiency analysis shows that $E(n)$ is 340 times higher than $E(\theta_0)$ and 6300 times higher than LoRA rank-8.

## Highlights & Insights
- **Theoretical Analysis of Parameter Efficiency**: ReNeg offers a theoretical explanation for why optimizing negative embeddings is more efficient than model fine-tuning, using a parameter efficiency metric based on the Jacobian matrix. This analytical framework itself can be extended to other parameter selection problems.
- **Plug-and-Play with Minimal Storage**: The learned negative embedding is merely an embedding vector resulting in near-zero storage costs, while yielding performance comparable to full-model fine-tuning, holding high practical value for the community.
- **Introducing CFG during Training**: This is a simple but overlooked key design that bridges a crucial inference-time component with the training-time gradient flow.

## Limitations & Future Work
- The global strategy uses the same embedding for all prompts, which may not be Pareto optimal.
- Sample-wise optimization requires additional inference-time computation (up to 10 optimization steps), sacrificing some inference speed.
- Training relies heavily on the HPSv2.1 reward model; the bias inherent in the reward model will propagate to the learned embeddings.
- Not validated on newer architectures like SDXL or Flux (which employ different dual text encoders, leaving transferability to be confirmed).

## Related Work & Insights
- **vs DNP**: DNP searches for negative prompts in a discrete language space, whereas ReNeg optimizes via gradients in a continuous embedding space, making it much more efficient.
- **vs ReFL/DDPO**: These methods fine-tune the entire UNet/diffusion model to enhance quality, whereas ReNeg achieves comparable or even superior results by optimizing only a single embedding vector.
- **vs TextCraftor**: TextCraftor fine-tunes the text encoder to improve the representation of positive prompts, while ReNeg focuses on the negative prompt. The two methods are orthogonal and can be combined.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The approach of integrating CFG into training to learn negative embeddings is novel and well-supported theoretically.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Conducts multi-benchmark evaluations, cross-model transfers, parameter efficiency theoretical analysis, and video generation transfers.
- Writing Quality: ⭐⭐⭐⭐ Features clear theoretical analysis and comprehensive experimental comparisons.
- Value: ⭐⭐⭐⭐⭐ A plug-and-play solution with minimal overhead, carrying high practical value for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] StyleKeeper: Prevent Content Leakage using Negative Visual Query Guidance](../../ICCV2025/image_generation/stylekeeper_prevent_content_leakage_using_negative_visual_query_guidance.md)
- [\[CVPR 2025\] Reward Fine-Tuning Two-Step Diffusion Models via Learning Differentiable Latent-Space Surrogate Reward](reward_fine-tuning_two-step_diffusion_models_via_learning_differentiable_latent-.md)
- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](../../NeurIPS2025/image_generation/training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [\[ICCV 2025\] REGEN: Learning Compact Video Embedding with (Re-)Generative Decoder](../../ICCV2025/image_generation/regen_learning_compact_video_embedding_with_re-generative_decoder.md)
- [\[CVPR 2025\] Rectified Diffusion Guidance for Conditional Generation](rectified_diffusion_guidance_for_conditional_generation.md)

</div>

<!-- RELATED:END -->
