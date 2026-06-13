---
title: >-
  [Paper Note] Identity-Preserving Image-to-Video Generation via Reward-Guided Optimization
description: >-
  [CVPR 2026][Video Generation][image-to-video] This paper proposes IPRO, which directly optimizes a video diffusion model via reinforcement learning and a differentiable facial identity scorer…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "image-to-video"
  - "identity preservation"
  - "reinforcement learning"
  - "face reward"
  - "diffusion model fine-tuning"
date: 2026-05-08
content_hash: 7f9cbc309cebc92a
---

# Identity-Preserving Image-to-Video Generation via Reward-Guided Optimization

**Conference**: CVPR 2026
**arXiv**: [2510.14255](https://arxiv.org/abs/2510.14255)  
**Code**: [https://ipro-alimama.github.io/](https://ipro-alimama.github.io/) (project page)  
**Area**: Diffusion Models / Video Generation
**Keywords**: image-to-video, identity preservation, reinforcement learning, face reward, diffusion model fine-tuning

## TL;DR

This paper proposes IPRO, which directly optimizes a video diffusion model via reinforcement learning and a differentiable facial identity scorer, significantly improving face identity consistency in image-to-video generation without modifying the model architecture, achieving 20%–45% FaceSim gains on Wan 2.2.

## Background & Motivation

**Background**: Image-to-video (I2V) generation has advanced considerably, with Diffusion Transformer models such as CogVideoX, HunyuanVideo, and Wan capable of synthesizing temporally coherent, high-quality videos from static images. Human-centric video generation represents a key application of I2V.

**Limitations of Prior Work**: Existing I2V models struggle to preserve the identity of input portraits in generated videos, especially under large facial expression changes or significant motion. This problem is further exacerbated when the face occupies only a small region of the image. As the number of frames increases, errors propagate across frames, causing gradual identity degradation and divergence from the initial frame.

**Key Challenge**: On one hand, identity information is fully encoded in the first frame and is not missing. On the other hand, existing approaches that inject additional identity modules into the model suffer from *exposure bias*—training relies on ground-truth intermediate states, while inference relies on the model's own generated states, leading to error accumulation and identity drift. Moreover, such architecture-invasive methods are inherently designed for single-person scenarios and do not generalize well to multi-person settings.

**Goal**: Can the identity-preserving capability of a general-purpose I2V foundation model be enhanced without modifying its architecture or compromising its original generation quality?

**Key Insight**: From a reinforcement learning perspective, a facial identity scorer (ArcFace) is used as a reward model, and diffusion model parameters are directly optimized via gradient backpropagation to generate videos with stronger identity consistency.

**Core Idea**: The cosine similarity of ArcFace face embeddings serves as a differentiable reward signal; truncated backpropagation through the denoising process is used to fine-tune the video diffusion model and improve identity preservation.

## Method

### Overall Architecture

IPRO takes initial noise $x_T$ and a conditioning image as inputs, performs a full $T$-step sampling through the video diffusion model to produce generated videos, decodes them to pixel space via a frozen VAE decoder, and scores them with a face reward model. The reward signal is backpropagated to the trainable parameters of the diffusion model. The framework comprises three core components: facial reward feedback learning, the Facial Scoring Mechanism (FSM), and KL divergence regularization.

### Key Designs

1. **Facial Reward Feedback Learning**:

    - **Function**: Directly optimizes the diffusion model to maximize facial identity consistency in generated videos.
    - **Mechanism**: The objective is $J(\theta) = \mathbb{E}_{x_T \sim N(0,I)}[R_{face}(sample(\theta, x_T))]$, i.e., maximizing the face reward of videos sampled from random noise. To reduce memory consumption and accelerate optimization, the DRaFT truncation strategy is adopted, backpropagating gradients only through the last $K$ steps ($K=4$): $\nabla_\theta R_{face}^K = \sum_{t=0}^{K} \frac{\partial R_{face}}{\partial x_t} \cdot \frac{\partial x_t}{\partial \theta}$. This is motivated by the observation that late-stage denoising steps have the greatest influence on appearance details.
    - **Design Motivation**: Compared to supervised fine-tuning (SFT), reward feedback learning generates from pure noise, aligning the training distribution with the inference distribution and directly eliminating exposure bias. Frame-level SFT losses cannot perceive gradual, small-magnitude identity drift, whereas a holistic reward can directly optimize long-term identity consistency.

2. **Facial Scoring Mechanism (FSM)**:

    - **Function**: Provides a robust, multi-angle face reward signal and prevents copy-paste behavior.
    - **Mechanism**: All face detections from the ground-truth video frames form a feature pool. For each generated frame $i$, the average cosine similarity with all ground-truth faces is computed: $s_i = \frac{1}{F}\sum_{j=1}^{F} \cos(\phi(\hat{x}_i), \phi(x_j))$. The final reward is the mean score across all generated frames. This encourages the generated subject to resemble the real subject across multiple viewpoints while permitting natural expression variation.
    - **Design Motivation**: Computing similarity only against the reference image causes the model to learn to rigidly replicate the first-frame expression (copy-paste), sacrificing expression diversity. Computing against temporally aligned GT frames yields a weak signal under SFT training. FSM provides a broad and informative reward.

3. **KL Divergence Regularization**:

    - **Function**: Stabilizes training and prevents reward hacking.
    - **Mechanism**: A multi-step KL divergence constraint is imposed on the reverse sampling trajectory: $D_{KL}(p_\theta(x_{0:T}) || p_{\theta_{ref}}(x_{0:T})) = \sum_{t=1}^{K} \omega_t' \|v_\theta(x_t, t) - v_{\theta_{ref}}(x_t, t)\|^2$, penalizing deviation in velocity predictions between the optimized model and the original reference model at each step.
    - **Design Motivation**: Optimizing with the face reward alone causes the model to over-exploit the reward model, producing videos with rigid expressions and limited motion. KL regularization constrains the deviation within a small range, preserving the model's original video generation capability.

### Loss & Training

The Adam optimizer is used with a learning rate of 2e-5, training for 100 steps with a batch size of 64. The truncated gradient step count is $K=4$, the face reward weight is 0.1, and the KL loss weight is 1. For Wan2.2 27B-A14B, only the low-noise expert component is trained. The distilled Wan2.2-Lightning variant (8 steps, no CFG) is used to improve training efficiency. Training data consists of 960p videos collected from the internet, retaining scenes with small faces (maximum face bounding box not exceeding 100×100 pixels).

## Key Experimental Results

### Main Results

| Method | FaceSim↑ | SC↑ | BC↑ | AQ↑ | IQ↑ | DD↑ |
|---|---|---|---|---|---|---|
| In-house I2V (15B) | 0.477 | 0.977 | 0.978 | 0.664 | 0.729 | 8.93 |
| + IPRO | **0.696** (+45.9%) | 0.981 | 0.981 | 0.664 | 0.726 | 8.31 |
| Wan 2.2 5B | 0.379 | 0.942 | 0.955 | 0.648 | 0.727 | 27.79 |
| + IPRO | **0.546** (+44.1%) | 0.946 | 0.956 | 0.649 | 0.724 | 27.26 |
| Wan 2.2 A14B | 0.578 | 0.951 | 0.971 | 0.659 | 0.727 | 19.45 |
| + IPRO | **0.694** (+20.1%) | 0.954 | 0.972 | 0.661 | 0.725 | 19.17 |

**Comparison with Other Methods (based on Wan 2.2 A14B)**:

| Method | FaceSim↑ |
|---|---|
| Wan 2.2 | 0.578 |
| MoCA† (adapted from T2V) | 0.582 |
| Concat-ID† (adapted from T2V) | 0.606 |
| DPO | 0.628 |
| GRPO | 0.633 |
| **IPRO (Ours)** | **0.694** |

### Ablation Study

| Configuration | FaceSim↑ | Hacking↓ | Notes |
|---|---|---|---|
| Wan 2.2 baseline | 0.578 | 7% | Baseline |
| w/o KL regularization | 0.754 | **58%** | High FaceSim but severe hacking |
| w/o FSM | 0.739 | **52%** | Similarly severe hacking |
| **Full IPRO** | **0.694** | **10%** | Balanced identity and natural motion |

| Training Framework | FaceSim↑ |
|---|---|
| SFT† | 0.639 |
| CLIP reward† | 0.610 |
| **IPRO (ArcFace reward)** | **0.694** |

### Key Findings

- KL regularization and FSM are critical for preventing reward hacking: removing either component leads to hacking rates exceeding 50%.
- ArcFace as a reward model substantially outperforms CLIP (0.694 vs. 0.610), owing to ArcFace's stronger discriminability for fine-grained facial features.
- Backpropagating through late (low-noise) denoising steps outperforms early (high-noise) steps: FaceSim 0.694 vs. 0.646.
- IPRO improves identity preservation while largely preserving the original model's video quality metrics.

## Highlights & Insights

- **Architecture-agnostic generality**: IPRO is a pure policy optimization method that requires no additional modules and can be directly applied to any I2V foundation model. This "reward-driven fine-tuning" paradigm is highly generalizable and converges in as few as 100 steps.
- **Multi-view pool design in FSM**: Using all ground-truth video frames as a reference pool rather than a single frame or a temporally aligned frame avoids copy-paste behavior while providing richer supervision—an elegant solution to the contradictory requirement of "maintaining consistency while allowing variation."
- **Quantitative analysis of KL regularization and reward hacking**: Employing Gemini 2.5 Pro VLM to quantitatively assess the hacking rate constitutes a compelling evaluation methodology.

## Limitations & Future Work

- The current method focuses solely on facial identity preservation; consistency of non-facial attributes (e.g., jewelry, accessories, clothing) is not addressed.
- Training relies on a dataset of small-face scenarios; improvements on large-face scenarios may be limited.
- Biases inherent in ArcFace (e.g., reduced recognition accuracy for certain ethnicities or extreme head poses) may be transferred to generated outputs.
- Future work could explore a unified "full-body identity" reward model covering both facial and non-facial features.

## Related Work & Insights

- **vs. MoCA / Concat-ID (T2V identity methods adapted to I2V)**: These methods require additional identity modules, altering the model architecture. IPRO achieves better performance without architectural modifications, suggesting that a correct optimization objective is more important than adding modules.
- **vs. DPO**: DPO optimizes relative preference rankings without absolute calibration; since facial identity is an absolutely quantifiable metric, the direct reward optimization in IPRO is better suited to this task.
- **vs. GRPO**: GRPO relies on intra-group response diversity for advantage estimation, but videos generated from the same prompt are highly similar, causing the advantage estimates to degenerate.

## Rating

- Novelty: ⭐⭐⭐⭐ First work to apply facial reward feedback learning to identity preservation in I2V generation; the FSM and KL regularization designs are elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across three foundation models with multiple baselines, detailed ablations, and a user study.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated; ablation experiments are logically structured.
- Value: ⭐⭐⭐⭐ Addresses an important practical problem in I2V generation with a generalizable and transferable methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Goal-Driven Reward by Video Diffusion Models for Reinforcement Learning](goal-driven_reward_by_video_diffusion_models_for_reinforcement_learning.md)
- [\[ICCV 2025\] DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization](../../ICCV2025/video_generation/dollar_fewstep_video_generation_via_distillation_and_latent.md)
- [\[ICCV 2025\] Multi-identity Human Image Animation with Structural Video Diffusion](../../ICCV2025/video_generation/multi-identity_human_image_animation_with_structural_video_diffusion.md)
- [\[CVPR 2026\] Anti-I2V: Safeguarding your photos from malicious image-to-video generation](anti-i2v_safeguarding_your_photos_from_malicious_image-to-video_generation.md)
- [\[CVPR 2026\] LocalDPO: Direct Localized Detail Preference Optimization for Video Diffusion Models](mind_the_generative_details_direct_localized_detail_preference_optimization_for_.md)

</div>

<!-- RELATED:END -->
