---
title: >-
  [Paper Note] Mimir: Improving Video Diffusion Models for Precise Text Understanding
description: >-
  [CVPR 2025][Video Generation][Text Understanding] Mimir proposes an end-to-end training framework that losslessly fuses the strong text understanding capabilities of a decoder-only LLM (Phi-3.5) with the stable features of a traditional text encoder (T5) through a meticulously designed Token Fuser. This significantly improves the text understanding accuracy of video diffusion models, achieving a substantial lead over existing methods, especially in multi-object…
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Text Understanding"
  - "Large Language Models"
  - "Token Fusion"
  - "Decoder-only LLM"
date: 2026-05-08
content_hash: de7b07901dbc6a37
---

# Mimir: Improving Video Diffusion Models for Precise Text Understanding

**Conference**: CVPR 2025  
**arXiv**: [2412.03085](https://arxiv.org/abs/2412.03085)  
**Code**: [https://lucaria-academy.github.io/Mimir/](https://lucaria-academy.github.io/Mimir/)  
**Area**: Video Generation / Diffusion Models  
**Keywords**: Video Generation, Text Understanding, Large Language Models, Token Fusion, Decoder-only LLM

## TL;DR
Mimir proposes an end-to-end training framework that losslessly fuses the strong text understanding capabilities of a decoder-only LLM (Phi-3.5) with the stable features of a traditional text encoder (T5) through a meticulously designed Token Fuser. This significantly improves the text understanding accuracy of video diffusion models, achieving a substantial lead over existing methods, especially in multi-object, spatial relation, and temporal understanding.

## Background & Motivation

1. **Background**: Current text-to-video (T2V) diffusion models typically use CLIP or T5 as text encoders. These encoder-based models have limited text understanding capabilities, making it difficult to precisely grasp fine-grained details such as complex spatial relations, quantities, colors, and temporal actions.
2. **Limitations of Prior Work**: Decoder-only LLMs (e.g., Phi-3.5, LLaMA) possess text understanding and reasoning capabilities far superior to encoders, but their feature distributions are incompatible with established T2V models: (1) massive scale differences in features (T5 features are concentrated in [-0.5, 0.5], while Phi-3.5 exceeds [-1, 1]); (2) feature volatility—decoder-only models may produce different features for multiple encodings of the same input (due to their generative nature), leading to training collapse if fused directly.
3. **Key Challenge**: How to leverage the reasoning capabilities of decoder-only LLMs while preserving the established video priors of T2V models, avoiding training instability caused by incompatible feature distributions.
4. **Goal**: Achieve lossless fusion of heterogeneous text features from encoders and decoder-only LLMs, allowing T2V models to simultaneously benefit from stable video priors and precise text understanding.
5. **Key Insight**: Rather than retraining the LLM as an encoder (which would destroy its reasoning capacity), design dedicated fusion modules that use Zero-Conv for progressive fusion and a semantic stabilizer to suppress feature volatility.
6. **Core Idea**: Bridge the heterogeneous features of the T5 encoder and the Phi-3.5 decoder-only LLM using a Token Fuser (Zero-Conv lossless fusion + Semantic Stabilizer semantic stabilization).

## Method

### Overall Architecture
Based on standard T2V diffusion models (3D Causal VAE + DiT), Mimir adds a decoder-only LLM branch. Input text is simultaneously sent to both the T5 encoder (obtaining $e_\theta$) and the Phi-3.5 decoder-only LLM (obtaining $e_\beta$ and instruction tokens $e_i$). The Token Fuser fuses the two pathways of features into a unified conditional embedding, which is fed into the DiT to guide video generation. It consists of two core components: non-destructive fusion and a semantic stabilizer.

### Key Designs

1. **Non-Destructive Fusion**:
    - **Function**: Fuses T5 encoder tokens and Phi-3.5 decoder tokens without destroying the original semantic space.
    - **Mechanism**: (1) Apply normalization and learnable scaling (Norm & Scale) to the decoder-only tokens $e_\beta$ to align their scale to the range of the encoder tokens; (2) Place a Zero-Conv layer $\mathcal{Z}_\beta$ after the decoder-only branch, yielding zero output at the start of training to avoid interfering with the original model; (3) Place a Zero-Conv layer $\mathcal{Z}_\theta$ (in a residual manner) after the encoder branch to ensure it equals the original encoder tokens at the beginning of training. The final fused condition is $e = e_\theta + \alpha \cdot e_\beta$, where $e_\theta = \tau_\theta(\mathcal{T}) + \mathcal{Z}_\theta(\tau_\theta(\mathcal{T}))$ and $e_\beta = \mathcal{Z}_\beta(\tau_\beta(\mathcal{T}))$.
    - **Design Motivation**: Direct addition leads to training collapse (in ablation studies, all metrics plummeted, e.g., Object Class dropped from 87.82% to 4.97%). Zero-Conv ensures that the newly added LLM branch contributes progressively starting from zero, serving as a classic "harmless initialization" strategy.

2. **Semantic Stabilizer**:
    - **Function**: Suppresses the volatility of decoder-only LLMs producing different features for the same input, while guiding the model to focus on key semantic elements.
    - **Mechanism**: Four attribute-specific instruction prompts (e.g., "describe objects in the video," "describe colors," etc.) are designed and input to the LLM to generate instruction tokens $e_i$. Additionally, four learnable anchor tokens $e_l$ are added as a bridge to the visual space: $e_s = e_i + e_l$. The $e_s$ is concatenated with the fused tokens $e$ and sent to the DiT. Instruction tokens provide distinct directions for semantic attention, while the learnable tokens are automatically adjusted during training to stabilize volatility.
    - **Design Motivation**: t-SNE visualization reveals that encoding the same prompt 50 times yields completely identical query tokens (a single point) but a wide distribution of answer tokens (due to the generative uncertainty of decoder-only models). Eliminating volatility entirely would sacrifice the reasoning advantages of LLMs. The Semantic Stabilizer restricts harmful volatility while retaining diversity.

3. **End-to-End Training Strategy**:
    - **Function**: Efficiently trains the joint T2V + LLM system.
    - **Mechanism**: Parameters of both T5 encoder and Phi-3.5 are frozen; only the Zero-Conv layers, learnable scaling parameters, and learnable anchor tokens are trained. The visual transformer parameters in DiT are trained normally. Training utilizes v-prediction with a zero SNR noise schedule.
    - **Design Motivation**: Freezing the two large language models significantly reduces training overhead, limiting trainable parameters to a small number within the fusion module.

### Loss & Training
- Standard diffusion loss: $\mathcal{L} = \mathbb{E}[\|\epsilon - \epsilon_\theta(z_t, e \oplus e_s, t)\|_2^2]$, where $e \oplus e_s$ represents the fused text condition.
- Phi-3.5 mini-instruct is used as the decoder-only LLM, with 500K high-quality video clips as training data.

## Key Experimental Results

### Main Results

**VBench Evaluation**:

| Method | Background Consistency | Aesthetic Quality | Object Class | Multiple Objects | Spatial Relation | Temporal Style |
|------|----------------------|-------------------|-------------|-----------------|-----------------|---------------|
| CogVideoX-5B | 95.60% | 60.62% | 87.82% | 65.70% | 64.86% | 25.86% |
| OpenSora | 97.20% | 58.57% | 90.79% | 64.81% | 76.63% | 25.51% |
| **Mimir** | **97.68%** | **62.92%** | **92.87%** | **85.29%** | **78.67%** | **26.22%** |

**User Study (10 participants, 20 prompt groups)**:

| Method | Instruction Following | Physics Simulation | Visual Quality |
|------|----------------------|-------------------|---------------|
| CogVideoX-5B | 72.15% | 57.30% | 63.25% |
| **Mimir** | **82.00%** | **83.65%** | **89.65%** |

### Ablation Study

| Configuration | Object Class | Multiple Objects | Spatial Relation |
|------|-------------|-----------------|-----------------|
| Baseline (T5 only) | 87.82% | 65.70% | 64.86% |
| + Decoder-only (Direct Addition) | 4.97% | 0.00% | 2.36% |
| + Decoder-only + Norm | 85.50% | 65.24% | 59.28% |
| + Decoder-only + ZeroConv | 92.03% | 84.98% | 69.17% |
| + ZeroConv + SS | 91.21% | 84.47% | 70.16% |
| **Mimir (All)** | **92.87%** | **85.29%** | **78.67%** |

### Key Findings
- Direct fusion of encoder and decoder-only tokens leads to training collapse (Object Class drops from 87% to 5%, and Multiple Objects drops to 0%), proving that incompatible feature distributions pose a severe issue.
- Zero-Conv is the most critical component: adding Zero-Conv alone improves Multiple Objects from 65.70% to 84.98% (+19.3%), far exceeding the effect of performing Normalization alone (65.24%).
- Semantic Stabilizer makes a significant contribution to spatial relation understanding: the complete version achieves 78.67% in Spatial Relation, compared to 69.17% without SS.
- Mimir consistently outperforms all compared methods across all VBench metrics, maintaining a massive lead in the user study regarding instruction following, physics simulation, and visual quality.
- t-SNE analysis clearly reveals the distribution gap between encoder and decoder-only features, as well as the volatility of decoder-only features, providing intuitive evidence for the design of the fusion strategy.

## Highlights & Insights
- **Zero-Conv progressive fusion strategy**: At the start of training, the contribution of the decoder-only branch is zero and gradually increases during training, avoiding training collapses caused by distribution conflicts. This is an elegant transfer of the ControlNet concept to text-condition fusion.
- **Instruction tokens + learnable anchors**: Leverages the instruction-following capability of LLMs to generate attribute-specific semantic guidance while utilizing learnable tokens as a bridge between visual and language spaces, complementing each other. This design can be generalized to any scenario requiring the fusion of heterogeneous features.
- **First integration of decoder-only LLM in video diffusion models**: Prior attempts were made only in the image generation domain (e.g., LiDiT, SANA), mostly using simple adapter schemes. Mimir's Token Fuser addresses the additional complexities introduced by temporal modeling in video generation.

## Limitations & Future Work
- Only Phi-3.5 mini (3.8B parameters) is used. Larger LLMs might yield better text understanding but at the cost of higher computational overhead, an exploration of which remains unaddressed.
- The training data consists of 500K videos, which is relatively limited in scale; larger datasets could further enhance performance.
- The four fixed instruction prompts are manually designed and might not be optimal; automatic search or adaptive instructions could be explored.
- The performance in image generation scenarios is not explored, although theoretically, the Token Fuser is equally applicable to T2I models.
- The choice of the number of learnable tokens (n=4) in the semantic stabilizer lacks in-depth analysis.

## Related Work & Insights
- **vs CogVideoX-5B**: A strong baseline in the current T2V domain, utilizing only T5 as the text encoder. Mimir outperforms it significantly in Multiple Objects (85.29% vs 65.70%), proving that the reasoning capabilities of LLMs are crucial for understanding complex scenes.
- **vs LiDiT/SANA**: These T2I methods also attempt to introduce LLMs, but LiDiT trains DiT from scratch and SANA uses a simple adapter; neither handles temporal distribution issues in video generation. Mimir's Zero-Conv + SS strategy is more robust.
- **vs ParaDiffusion/LaVi-Bridge**: These methods bridge Phi3 and PixArt with an adapter. The authors show that simple adapters do not perform well in video generation, making the design of Token Fuser much more comprehensive.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of Token Fuser has engineering innovation value, though Zero-Conv itself is not a new concept; the primary contribution lies in the clever combination of existing techniques to address a new problem.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Broad evaluation on VBench + user study + detailed ablation studies + t-SNE visualization analysis make the experimental design comprehensive and thorough.
- Writing Quality: ⭐⭐⭐⭐ The motivation is clearly articulated, and the core concepts and analysis plots in Figures 2 and 7 are highly convincing.
- Value: ⭐⭐⭐⭐ Provides a feasible path for incorporating LLMs into T2V models; the Token Fuser can be reused in other multi-source feature fusion scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VideoDirector: Precise Video Editing via Text-to-Video Models](videodirector_precise_video_editing_via_text-to-video_models.md)
- [\[CVPR 2025\] VideoGuide: Improving Video Diffusion Models without Training Through a Teacher's Guide](videoguide_improving_video_diffusion_models_without_training_through_a_teachers_.md)
- [\[CVPR 2025\] Towards Precise Scaling Laws for Video Diffusion Transformers](towards_precise_scaling_laws_for_video_diffusion_transformers.md)
- [\[CVPR 2025\] ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models](shotadapter_text-to-multi-shot_video_generation_with_diffusion_models.md)
- [\[ICLR 2026\] Improving Autoregressive Video Modeling with History Understanding](../../ICLR2026/video_generation/improving_autoregressive_video_modeling_with_history_understanding.md)

</div>

<!-- RELATED:END -->
