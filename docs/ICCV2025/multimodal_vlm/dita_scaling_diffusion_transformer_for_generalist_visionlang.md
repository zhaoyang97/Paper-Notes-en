---
title: >-
  [Paper Note] Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy
description: >-
  [ICCV 2025][Multimodal VLM][VLA] This paper proposes Dita (Diffusion Transformer Policy), which, unlike prior methods that denoise on compressed embeddings using shallow networks, adopts in-context conditioning to directly condition denoising on raw visual tokens. A causal Transformer processes the full token sequence of language, images, timesteps, and noisy actions. With 334M parameters, Dita achieves state-of-the-art or competitive performance on SimplerEnv zero-shot, LIBERO, CALVIN, and other benchmarks.
tags:
  - ICCV 2025
  - Multimodal VLM
  - VLA
  - diffusion policy
  - DiT
  - in-context conditioning
  - cross-embodiment
date: 2026-05-08
content_hash: ae48ec0ddfed1d42
---

# Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy

**Conference**: ICCV 2025
**arXiv**: [2503.19757](https://arxiv.org/abs/2503.19757)
**Code**: [Project](https://robodita.github.io)
**Area**: Robot Policy / VLA Models
**Keywords**: VLA, diffusion policy, DiT, in-context conditioning, cross-embodiment

## TL;DR

This paper proposes Dita (Diffusion Transformer Policy), which, unlike prior methods that denoise on compressed embeddings using shallow networks, adopts in-context conditioning to directly condition denoising on raw visual tokens. A causal Transformer processes the full token sequence of language, images, timesteps, and noisy actions. With 334M parameters, Dita achieves state-of-the-art or competitive performance on SimplerEnv zero-shot, LIBERO, CALVIN, and other benchmarks.

## Background & Motivation

- **Background**: Generalist robot policies have advanced through pretraining on large-scale cross-embodiment datasets such as OXE.
- **Limitations of Prior Work**: (1) Discretized actions (e.g., OpenVLA) limit adaptability to heterogeneous action spaces; (2) methods using MLP or small DiT diffusion heads (e.g., Octo, π₀) lack sufficient expressiveness under the diversity of large-scale data; (3) denoising on embeddings discards visual details from historical observations.
- **Key Challenge**: Heterogeneous cross-embodiment action spaces conflict with the need for a unified policy representation.
- **Goal**: Design an expressive, scalable generalist robot policy architecture.
- **Key Insight**: Place action denoising directly inside a causal Transformer that interacts with visual tokens.
- **Core Idea**: Action denoising should not operate on compressed embeddings but should instead perform in-context attention directly over raw visual patch tokens.

## Method

### Overall Architecture

CLIP encodes language → DINOv2 + Q-Former extract image features → concatenate [language, image, timestep, noisy action] token sequence → causal DiT denoising → output clean action chunk (16 steps).

### Key Designs

**Design 1: In-Context Conditioning Diffusion**
- **Function**: Noisy action tokens and visual/language tokens are processed together within a single causal Transformer.
- **Mechanism**: Action tokens participate directly in attention computation and can attend to every image patch token, capturing subtle action increments and environmental details.
- **Design Motivation**: Prior methods condition denoising on a single embedding, losing spatial detail; in-context conditioning preserves complete visual information.

**Design 2: End-to-End DINOv2 Fine-Tuning + Q-Former**
- **Function**: DINOv2 extracts multi-scale features; Q-Former queries key visual features conditioned on language instructions.
- **Mechanism**: DINOv2 pretrained on web data has a domain gap with robot data; end-to-end fine-tuning bridges this gap. Q-Former uses FiLM conditioning to select task-relevant information from DINOv2 patch features, reducing computational cost.
- **Design Motivation**: Frozen visual encoders are insufficient for robotics, but full fine-tuning produces too many tokens, necessitating Q-Former compression.

**Design 3: Lightweight and Scalable Architecture**
- **Function**: Achieves state-of-the-art performance with only 334M parameters.
- **Mechanism**: LLaMA-style causal Transformer, requiring no large VLM (e.g., PaliGemma). DDPM training (1000 steps) + DDIM inference (20 steps).
- **Design Motivation**: Provides a clean, lightweight, open-source baseline that lowers the barrier to entry for the community.

### Loss & Training

Standard DDPM diffusion objective: $\min \|\epsilon - \epsilon_\theta(z_t, t, c)\|^2$. AdamW optimizer, 100K steps, batch size 8192 (32× A100), 2-frame observation history → 16-step action chunk.

## Key Experimental Results

### Main Results

**SimplerEnv Zero-Shot (Success Rate %)**

| Method | coke_can (match/var) | move_near (match/var) |
|--------|----------------------|-----------------------|
| RT-1-X | 56.7 / 49.0 | 31.7 / 32.3 |
| OpenVLA | 16.3 / 54.5 | 46.2 / 47.7 |
| **Dita** | **83.7 / 85.5** | **76.0 / 73.0** |

**LIBERO Fine-Tuning (Success Rate %)**

| Method | SPATIAL | OBJECT | GOAL | LONG | Avg. |
|--------|---------|--------|------|------|------|
| OpenVLA | 84.9 | 88.4 | 79.2 | 53.7 | 76.5 |
| **Dita** | 84.2 | **96.3** | **85.4** | **63.8** | **82.4** |

### Ablation Study

| Configuration | Calvin Avg. Len |
|---------------|----------------|
| Diffusion head (non-in-context) | 3.16 |
| In-context Dita | **3.53** |
| No pretraining | 2.38 |

### Key Findings

1. In-context conditioning significantly outperforms diffusion heads, especially on long-horizon tasks (LIBERO-LONG +10%).
2. Using only a third-person camera with 10-shot fine-tuning generalizes to novel real-world environments.
3. 334M parameters surpass 7B (OpenVLA) and larger models, suggesting that architectural design matters more than scale.

## Highlights & Insights

1. The core insight of in-context conditioning — action denoising requires access to raw visual details rather than compressed embeddings.
2. The lightweight open-source baseline provides significant value to the community.
3. The paradigm of cross-embodiment pretraining followed by 10-shot real-world fine-tuning is highly practical.

## Limitations & Future Work

1. Only third-person cameras are used; incorporating wrist cameras or tactile sensing could further improve performance.
2. The sensitivity of Q-Former query count to performance is not thoroughly analyzed.
3. Validation on bimanual manipulation scenarios is absent.

## Related Work & Insights

- Octo employs a diffusion head but with limited expressiveness; Dita demonstrates that internalizing denoising within the Transformer is superior.
- π₀ relies on a larger VLM, yet Dita achieves comparable performance with only 334M parameters.
- Insight: The key to robot policy learning may lie not in model size but in how actions and observations interact.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ★★★★☆ |
| Practicality | ★★★★★ |
| Experimental Thoroughness | ★★★★★ |
| Writing Quality | ★★★★☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CoA-VLA: Improving Vision-Language-Action Models via Visual-Textual Chain-of-Affordance](coavla_improving_visionlanguageaction_models_via_visualtext.md)
- [\[ICCV 2025\] MaTVLM: Hybrid Mamba-Transformer for Efficient Vision-Language Modeling](matvlm_hybrid_mamba-transformer_for_efficient_vision-language_modeling.md)
- [\[ICCV 2025\] TAB: Transformer Attention Bottlenecks enable User Intervention and Debugging in Vision-Language Models](tab_transformer_attention_bottlenecks_enable_user_intervention_and_debugging_in_.md)
- [\[ICCV 2025\] Chimera: Improving Generalist Model with Domain-Specific Experts](chimera_improving_generalist_model_with_domain-specific_experts.md)
- [\[ICCV 2025\] Scaling Inference-Time Search with Vision Value Model for Improved Visual Comprehension](scaling_inferencetime_search_with_vision_value_model_for_imp.md)

</div>

<!-- RELATED:END -->
