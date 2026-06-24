---
title: >-
  [Paper Note] Learning Trimodal Relation for Audio-Visual Question Answering with Missing Modality
description: >-
  [ECCV 2024][Image Generation] A missing-modality AVQA framework based on trimodal relations is proposed. It recalls missing modality features via the RMM generator and enhances them cross-modally using the AVR diffusion model, achieving accurate question answering even when audio or visual modality is missing.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: 3ed054c3e5095d0e
---

# Learning Trimodal Relation for Audio-Visual Question Answering with Missing Modality

**Conference**: ECCV 2024  
**arXiv**: [2407.16171](https://arxiv.org/abs/2407.16171)  
**Area**: Image Generation

## TL;DR

A missing-modality AVQA framework based on trimodal relations is proposed. It recalls missing modality features via the RMM generator and enhances them cross-modally using the AVR diffusion model, achieving accurate question answering even when audio or visual modality is missing.

## Background & Motivation

Existing audio-visual question answering (AVQA) methods rely on complete visual and audio inputs. However, in real-world scenarios, device failures or data transmission errors often lead to the absence of a certain modality, severely degrading the performance of existing methods. Most prior missing-modality methods process one-to-one modality pairs, ignoring the mutual dependencies among different modalities, and especially failing to flexibly generate pseudo-features adaptive to the question context.

Inspired by human cognitive psychology—where humans can recall missing details through audio-visual association—this paper proposes a novel AVQA framework to address the missing modality issue.

## Method

### Overall Architecture

The system consists of three core components:
1. **Relation-aware Missing Modal (RMM) Generator**: Utilizes the two available modalities (e.g., visual + text) to recall the pseudo-features of the missing modality (e.g., audio).
2. **Audio-Visual Relation-aware (AVR) Diffusion Model**: Concatenates pseudo-features with real features and enhances feature representations cross-modally through a diffusion process.
3. **AVQA Backbone Network**: Receives the enhanced features for question-answering prediction.

### Key Designs

The **RMM Generator** adopts a slot-based architecture, where each modality is represented by $L$ learnable parameter vectors. Utilizing an addressing vector mechanism, it calculates the correlation between the available modality features and the slots, fuses the visual-text addressing vectors through element-wise multiplication, and obtains the pseudo-features by computing a weighted sum of the missing modality's slots. The generators for the three modalities share weights.

The **AVR Diffusion Model** concatenates audio and visual features to create joint features. Through forward noise addition and reverse denoising processes, it learns to leverage cross-modal complementary information to enhance feature representations. Real feature pairs are used during training, while combinations of pseudo-features and real features are used during inference.

### Loss & Training

The total loss function consists of three components:

$$\mathcal{L}_{Total} = \mathcal{L}_{avqa} + \lambda_1 \mathcal{L}_{rmmr} + \lambda_2 \mathcal{L}_{ave}$$

- $\mathcal{L}_{rmmr}$: Relation-aware missing modality recall loss, which imposes an $L_2$ constraint to guide pseudo-features to approximate real features.
- $\mathcal{L}_{ave}$: Audio-visual enhancement loss, formulated as a standard diffusion denoising loss.
- $\mathcal{L}_{avqa}$: Three sets of cross-entropy losses (complete input, missing audio, and missing visual).

Hyperparameters are set to $\lambda_1 = \lambda_2 = 1$, the number of RMM slots is $L=75$, and the number of diffusion steps is $T=10$.

## Key Experimental Results

### Main Results

Results of various AVQA networks integrated with Ours on the MUSIC-AVQA dataset (All Avg accuracy %):

| Method | Missing Visual (Original) | Missing Visual (+Ours) | Missing Audio (Original) | Missing Audio (+Ours) |
|------|:-:|:-:|:-:|:-:|
| AVSD | 59.25 | **68.91** | 41.08 | **69.90** |
| Pano-AVQA | 51.14 | **67.87** | 42.91 | **69.90** |
| AVST | 59.14 | **67.98** | 36.60 | **69.71** |
| PSTP-Net* | 59.27 | **66.39** | 67.74 | **71.55** |

The improvement in the missing audio scenario is particularly significant, with AVST increasing from 36.60% to 69.71% (+33.11%).

### Ablation Study

Comparison with other missing-modality handling methods on MUSIC-AVQA (based on the AVST backbone, All Avg %):

| Method | Missing Visual | Missing Audio |
|------|:-:|:-:|
| ActionMAE | 64.12 | 65.38 |
| ShaSpec | 63.87 | 66.21 |
| Missing-aware Prompting | 65.44 | 67.13 |
| **Ours** | **67.98** | **69.71** |

### Key Findings

- The performance gain is larger in the missing audio scenario, showing that the visual modality contains more information that can be leveraged for cross-modal recall.
- The RMM generator, by utilizing trimodal relations, is more effective than approaches based on single modality pairs.
- The AVR diffusion model further enhances the representation quality of both pseudo-features and real features.
- The proposed method can be flexibly integrated into various AVQA backbone networks.

## Highlights & Insights

- First to simultaneously address bidirectional missing modalities (both missing audio and missing visual) in the AVQA task.
- The slot-based addressing mechanism elegantly leverages trimodal relations to generate pseudo-features.
- Employing a diffusion model for feature enhancement rather than image generation is a creative application.
- The general framework is plug-and-play for existing AVQA networks, offering high practicality.

## Limitations & Future Work

- It only handles the absence of a single modality, without considering extreme scenarios where multiple modalities are missing simultaneously.
- The diffusion process increases the inference time overhead.
- The quality of pseudo-features is affected by the number of slots $L$ and the distribution of training data.

## Rating

⭐⭐⭐⭐ High novelty, thorough experimentation, and substantial practical value in missing-modality scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Mutual Learning for Acoustic Matching and Dereverberation via Visual Scene-driven Diffusion](mutual_learning_for_acoustic_matching_and_dereverberation_via_visual_scene-drive.md)
- [\[ECCV 2024\] WebRPG: Automatic Web Rendering Parameters Generation for Visual Presentation](webrpg_automatic_web_rendering_parameters_generation_for_visual_presentation.md)
- [\[ECCV 2024\] Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)
- [\[ECCV 2024\] Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction](learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)
- [\[ECCV 2024\] MotionChain: Conversational Motion Controllers via Multimodal Prompts](motionchain_conversational_motion_controllers_via_multimodal_prompts.md)

</div>

<!-- RELATED:END -->
