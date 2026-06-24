---
title: >-
  [Paper Note] Optical-Flow Guided Prompt Optimization for Coherent Video Generation
description: >-
  [CVPR 2025][Video Generation][Video Diffusion Models] This paper proposes MotionPrompt, a training-free inference-time guidance method for video diffusion models. By optimizing learnable token embeddings in combination with an optical flow discriminator, it enhances the temporal consistency and motion smoothness of video generation.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Video Diffusion Models"
  - "Temporal Consistency"
  - "Optical Flow Guidance"
  - "Prompt Optimization"
  - "Training-free"
date: 2026-05-08
content_hash: b04c1f86fc4f9b49
---

# Optical-Flow Guided Prompt Optimization for Coherent Video Generation

**Conference**: CVPR 2025  
**arXiv**: [2411.15540](https://arxiv.org/abs/2411.15540)  
**Code**: [motionprompt.github.io](https://motionprompt.github.io/)  
**Area**: Diffusion Models / Video Generation  
**Keywords**: Video Diffusion Models, Temporal Consistency, Optical Flow Guidance, Prompt Optimization, Training-free

## TL;DR

This paper proposes MotionPrompt, a training-free inference-time guidance method for video diffusion models. By optimizing learnable token embeddings in combination with an optical flow discriminator, it enhances the temporal consistency and motion smoothness of video generation.

## Background & Motivation

**Background**: Text-to-video (T2V) diffusion models (such as VideoCrafter2, AnimateDiff, Lavie) have made significant progress in recent years, enabling the generation of visually rich videos based on text prompts. However, these models still suffer from obvious flaws in temporal consistency, manifested as object flickering, sudden appearance/disappearance, and color inconsistency.

**Limitations of Prior Work**: Under the diffusion model framework, guidance techniques have proven effective in improving output quality. However, applying them to video diffusion models presents unique challenges: (1) directly guiding latent representations requires backpropagation across all frames, which is computationally expensive and prone to instability; (2) fine-tuning video models is highly costly due to their massive scale; (3) providing guidance only to a subset of frames may disrupt inter-frame consistency.

**Key Challenge**: Video diffusion models require cross-frame temporal consistency guidance, but existing guidance methods either demand expensive gradient computations over all frames or necessitate additional model fine-tuning, lacking a lightweight and universal inference-time guidance mechanism.

**Goal**: Design a computationally efficient inference-time guidance method that enhances the temporal consistency of arbitrary text-to-video diffusion models without disrupting content fidelity or requiring model retraining.

**Key Insight**: The authors observe that text prompts can simultaneously influence all frames. Therefore, dynamically optimizing the prompt embeddings during the inference process allows indirect control over the entire video at a fraction of the computational cost of direct latent guidance. Combining optical flow as a signal for temporal consistency, a lightweight discriminator is trained to distinguish between the optical flow patterns of real and generated videos.

**Core Idea**: Append learnable tokens to the original prompt during inference and optimize their embeddings using gradients from an optical flow discriminator, thereby indirectly guiding the video diffusion model to generate temporally more consistent videos.

## Method

The core mechanism of MotionPrompt is "freeze the model, target only the prompt". During each reverse sampling step, it appends learnable tokens to the text prompt, evaluates the motion realism between randomly selected pairs of generated frames using an optical flow discriminator, and backpropagates the gradients to optimize these token embeddings, steering the video generation towards more natural motion patterns.

### Overall Architecture

The input consists of a text prompt and a pretrained video diffusion model. During the reverse diffusion sampling process, the following pipeline is executed at each timestep $t$: (1) append learnable tokens $S$ to the end of the original prompt $P$; (2) perform denoising with the current embeddings to obtain Tweedie-predicted clean frames; (3) randomly select frame pairs, decode them to pixel space, and compute their optical flow; (4) feed the optical flow into the discriminator to obtain the loss; (5) backpropagate gradients to update the token embeddings, incorporating TV regularization and embedding regularization; (6) repeat this optimization for $K$ iterations, and then perform the actual reverse sampling step using the optimized embeddings. In the latter half of the sampling steps, the original prompt is restored to preserve the overall appearance.

### Key Designs

1. **Learnable Token Embedding Optimization**:

    - **Function**: Indirectly guide video generation during inference by optimizing the appended token embeddings.
    - **Mechanism**: Append $n$ learnable tokens $S = \{S_i\}_{i=1}^n$ (initialized with words related to video quality, such as "authentic") to the original prompt $P$. During optimization, only the embeddings of $S$ are updated, while keeping the original prompt embeddings fixed. The optimization objective is $\hat{\mathcal{T}}_t = \arg\min_{\mathcal{T}} \ell(z_t, c(\mathcal{T}))$, where $\mathcal{T}$ represents the embeddings of $S$. Since text embeddings influence all frames simultaneously via cross-attention, optimizing the prompt acts as a low-dimensional proxy to indirectly control the high-dimensional video latent space.
    - **Design Motivation**: Computing gradients directly on video latent variables requires backpropagation across all frames, which is computationally expensive and potentially unstable. In contrast, prompt embeddings have a much lower dimensionality than the video latent space, making optimization far cheaper while globally affecting all frames. Keeping the original prompt embeddings intact ensures that the semantic content does not deviate.

2. **Optical Flow Discriminator $\phi_d$**:

    - **Function**: Evaluate whether the inter-frame optical flow conforms to the motion patterns of real videos, serving as a measure of temporal consistency.
    - **Mechanism**: Train a lightweight ViT-based discriminator that takes the optical flow field between two frames (extracted by RAFT) as input and outputs a "real/generated" probability. The training data consists of optical flows from real videos in DAVIS and WebVid, along with those from videos generated by various models. During inference, frame pairs are randomly selected from the Tweedie-predicted clean frames, and their computed optical flow is fed into the discriminator. The loss is defined as $\ell_{disc} = \log(1 - \phi_{\theta^*}(f))$, driving the optical flow of the generated frames to be recognized as "real" by the discriminator.
    - **Design Motivation**: Optical flow is a direct signal for measuring inter-frame motion consistency. Compared to pixel-level consistency constraints, optical flow focuses more on the naturalness and smoothness of motion. Using a discriminator rather than a reference optical flow allows adaptation to diverse motion patterns instead of a fixed pattern. Since discrimination is only performed on frame pairs, the computational overhead is significantly lower than analyzing the entire sequence.

3. **TV Regularization and Embedding Regularization**:

    - **Function**: Ensure the smoothness of the optical flow field and prevent the embeddings from drifting too far.
    - **Mechanism**: The total loss function is $\ell_{total} = \lambda_1 \ell_{disc} + \lambda_2 \ell_{TV} + \lambda_3 \|\mathcal{T} - \mathcal{T}_0\|_2^2$. The TV loss constrains the spatial smoothness of the optical flow field to prevent unnatural local motion mutations. The embedding $L_2$ regularization restricts the optimized token embeddings from deviating too far from their initialization, keeping them within the valid space of the text encoder.
    - **Design Motivation**: The optical flow field itself should be spatially smooth, and the TV loss aligns with this physical prior. The embedding regularization prevents the optimization process from shifting into unfamiliar regions of the text space, avoiding degradation in generation quality.

### Loss & Training

The discriminator is pretrained for approximately 20 epochs using the standard GAN discriminator loss, based on a pretrained ViT and a 3-layer MLP classifier. During inference, $K$ validation/optimization iterations are executed per sampling step. The method is robust to hyperparameter choices, consistently outperforming the baseline across all configurations.

## Key Experimental Results

### Main Results

| Model | Subject Consistency ↑ | Background Consistency ↑ | Temporal Flickering ↑ | Motion Smoothness ↑ |
|------|----------------------|-------------------------|---------------------|-------------------|
| AnimateDiff | 0.9488 | 0.9755 | 0.9228 | 0.9578 |
| + MotionPrompt | **0.9528** | 0.9763 | **0.9258** | **0.9599** |
| Lavie | 0.9599 | 0.9739 | 0.9487 | 0.9690 |
| + MotionPrompt | **0.9646** | **0.9781** | **0.9625** | **0.9765** |
| VideoCrafter2 | 0.9736 | 0.9559 | 0.9559 | 0.9750 |
| + MotionPrompt | **0.9745** | **0.9774** | **0.9588** | **0.9759** |

### Ablation Study

| Configuration | Effect |
|------|------|
| TV loss only (no discriminator) | Smoother motion, but limited overall consistency improvement |
| Discriminator only (no TV) | Greater consistency improvement, but may introduce minor local jitter |
| Discriminator + TV + Embedding regularization | Best balance |
| Without Embedding regularization | Degradation in some metrics, occasional decline in video quality |

### Key Findings

- User studies show that MotionPrompt is preferred by the majority of users across all three models (66.5% win rate for AnimateDiff, 55.1% win rate for Lavie, and 53.0% win rate for VideoCrafter2).
- The method significantly improves temporal quality while maintaining text alignment.
- The Dynamic Degree metric shows a slight decrease, indicating a trade-off between consistency and dynamics; however, visual results show a good balance is struck.
- Discriminator training and inference can be executed on a single A100 GPU.

## Highlights & Insights

- **Extremely lightweight**: Modifying only a few appended token embeddings without retraining the video models, resulting in minimal computational overhead.
- **High generalization**: Applicable to various video models with different architectures, such as Lavie, AnimateDiff, and VideoCrafter2.
- **Elegant combination of optical flow & discriminator**: Optical flow provides a natural signal to measure motion realism, while the discriminator eliminates the need for reference flows.
- **New paradigm of prompt optimization for video guidance**: The first to apply inference-time prompt optimization to video diffusion models, opening up a new research direction.

## Limitations & Future Work

- An inherent trade-off exists between temporal consistency and motion dynamics, currently relying on hyperparameter tuning to find a balance.
- The discriminator needs to be trained separately for each video model; cross-model transferability has not yet been validated.
- Validation is currently limited to 16-frame short videos; longer video scenarios require further exploration.
- Inference speed is reduced because each step requires multiple forward passes for prompt optimization.
- Future research can explore stronger motion quality signals, such as human motion plausibility and physical consistency.

## Related Work & Insights

- **MinorityPrompt**: The prompt optimization concept is inspired by MinorityPrompt, which utilizes learnable tokens to generate minority-class images. This work extends the idea to video temporal consistency.
- **DPS (Diffusion Posterior Sampling)**: A classic guidance method for diffusion models, but computationally prohibitive for video generation.
- **FreeInit**: Enhances consistency by refining the low-frequency information of initial noise, but is computationally expensive and may lose structural details.
- **Insight**: Prompt optimization is an under-explored inference-time control mechanism. Its low-dimensional nature is inherently suited for high-dimensional generation tasks such as video synthesis.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to utilize prompt optimization for temporal guidance in video diffusion models.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation involving three models, quantitative metrics, user studies, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear methodology description with logical flow.
- Value: ⭐⭐⭐⭐ — Highly practical and lightweight universal solution for video quality enhancement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] The Devil is in the Prompts: Retrieval-Augmented Prompt Optimization for Text-to-Video Generation](the_devil_is_in_the_prompts_retrieval-augmented_prompt_optimization_for_text-to-.md)
- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](../../ICCV2025/video_generation/vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[CVPR 2025\] VIRES: Video Instance Repainting via Sketch and Text Guided Generation](vires_video_instance_repainting_via_sketch_and_text_guided_generation.md)
- [\[CVPR 2025\] Geometry-guided Online 3D Video Synthesis with Multi-View Temporal Consistency](geometry-guided_online_3d_video_synthesis_with_multi-view_temporal_consistency.md)
- [\[ICCV 2025\] Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM](../../ICCV2025/video_generation/prompt-a-video_prompt_your_video_diffusion_model_via_preference-aligned_llm.md)

</div>

<!-- RELATED:END -->
