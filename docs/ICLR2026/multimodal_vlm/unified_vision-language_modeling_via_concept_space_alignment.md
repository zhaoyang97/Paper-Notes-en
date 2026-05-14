---
title: >-
  [Paper Note] Unified Vision-Language Modeling via Concept Space Alignment
description: >-
  [ICLR 2026][Multimodal VLM][Vision-language embedding space] This paper proposes v-Sonar, which post-hoc aligns a visual encoder to the SONAR text embedding space…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Vision-language embedding space"
  - "latent diffusion model"
  - "multilingual"
  - "video captioning"
  - "Large Concept Model"
date: 2026-05-08
content_hash: 9c1eee2ee693654b
---

# Unified Vision-Language Modeling via Concept Space Alignment

**Conference**: ICLR 2026
**arXiv**: [2603.01096](https://arxiv.org/abs/2603.01096)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: Vision-language embedding space, latent diffusion model, multilingual, video captioning, Large Concept Model

## TL;DR

This paper proposes v-Sonar, which post-hoc aligns a visual encoder to the SONAR text embedding space, enabling the Large Concept Model (LCM) trained in the SONAR space to handle visual inputs in a zero-shot manner. Through instruction fine-tuning, v-Sonar is extended into v-LCM, which surpasses existing VLMs in 61 out of 62 languages.

## Background & Motivation

Existing language- and modality-agnostic embedding spaces (e.g., SONAR, supporting 1,500 text languages and 177 speech languages) have achieved strong performance on text and speech tasks, but remain limited to those modalities and cannot handle visual inputs. The Large Concept Model (LCM) performs next-embedding prediction in the SONAR space using a diffusion objective, demonstrating the viability of language modeling in continuous embedding spaces rather than over discrete tokens.

The core motivation of this paper is twofold: (1) Can the visual modality be aligned to the SONAR space so that LCM can understand visual inputs without any visual training data? (2) Can LCM be further enhanced through visual-language instruction fine-tuning?

## Method

### Overall Architecture

The framework consists of three components: (1) v-Sonar: aligning the Perception Encoder (PE) to the SONAR text space; (2) zero-shot visual understanding validation with LCM; and (3) v-LCM: visual-language instruction fine-tuning in the unified v-Sonar + SONAR space.

### Key Designs

1. **v-Sonar Visual Encoder Alignment**: A lightweight projector is stacked on top of the Perception Encoder (PE). Positional encodings are first injected to provide temporal information across frames, followed by a temporal attention layer for inter-frame interaction, and finally attention pooling to aggregate all frames into a single video-level representation. The training objective minimizes the MSE between visual embeddings and SONAR text embeddings: $\mathcal{L}_{\text{align}} = \frac{1}{N}\sum_{i=1}^{N}\|f_\theta(V_i) - g(T_i)\|_2^2$, where the SONAR encoder $g$ is frozen and only the projector and visual encoder are updated.

2. **Coarse-to-Fine Curriculum Training**: A three-stage alignment strategy — Stage 1 uses 12M large-scale image-text pairs to establish a foundational mapping; Stage 2 introduces 2M synthetic video caption data to adapt to temporal dynamics; Stage 3 uses 200K high-quality human-annotated video captions for fine-grained alignment.

3. **v-LCM Latent Diffusion Visual-Language Model**: Visual embeddings (v-Sonar encoded) and text embeddings (SONAR encoded) are unified into a sequence of latent embeddings and trained with the same latent diffusion objective as LCM text pre-training. A two-tower architecture is adopted: a contextualizer encodes preceding embeddings, and a denoiser iteratively reconstructs the next embedding. The diffusion process is defined as $x_t = \alpha_t x^0 + \sigma_t \epsilon$, with training loss $\mathcal{L}(\theta) = \mathbb{E}\|x^0 - \mu_\theta(\alpha_t x^0 + \sigma_t \epsilon, t, c)\|_2$.

### Loss & Training

- v-Sonar alignment: MSE loss + coarse-to-fine three-stage curriculum
- Asynchronous learning rates are applied to the projector and encoder separately to prevent gradient instability
- Normalized initialization and attention pooling further improve alignment quality
- v-LCM: fine-tuned on M3IT multimodal multilingual instruction data using the same latent diffusion objective as LCM text pre-training

## Key Experimental Results

### Main Results

| Dataset | Metric | v-Sonar | PECoreG | SigLIP2-G-OPT |
|---------|--------|---------|---------|---------------|
| PE-Video | R@1 | **73.03** | 63.91 | 47.55 |
| Vatex | R@1 | **40.75** | 18.90 | 27.52 |
| Dream-1k | R@1 | 63.30 | **72.10** | 61.50 |

| Dataset | Metric | v-Sonar + OmniSONAR Decoder | PLM-3B | Qwen2.5-VL-3B |
|---------|--------|-----------------------------|--------|---------------|
| PE-Video | Bleu | **39.0** | 21.1 | 30.0 |
| Dream-1k | Bleu | **23.9** | 19.6 | 16.1 |
| Vatex-zh | R-L | **26.9** | - | - |

| M3IT Multilingual Evaluation | v-LCM | InternVL | Qwen-VL |
|-----------------------------|-------|----------|---------|
| Languages outperformed (out of 62) | **61/62** | - | - |

### Ablation Study

| Configuration | MSE↓ | Cos.Sim↑ | Bleu↑ | Notes |
|---------------|------|----------|-------|-------|
| Linear Proj. | 1.45e-3 | 0.694 | 38.0 | Frozen PE baseline |
| Full PE | 1.54e-3 | 0.672 | 37.1 | Full fine-tuning performs worse |
| + Async. LR | 1.43e-3 | 0.700 | 39.7 | Asynchronous learning rate is effective |
| + Norm. Init. | 1.39e-3 | 0.708 | 39.8 | Normalized initialization |
| + Attn. Pooling | 1.39e-3 | 0.708 | 39.8 | Attention pooling |
| Full Pipeline (3-stage) | **1.36e-3** | **0.716** | **40.1** | Best with complete three stages |
| w/o Stage 2 (SV) | 1.39e-3 | 0.710 | 39.6 | Without synthetic video stage |
| w/o Stage 1 & 2 | 1.39e-3 | 0.708 | 39.8 | Human annotations only |

### Key Findings

- v-Sonar improves retrieval R@1 over the original PE by 9.12 and 21.85 points on PE-Video and Vatex, respectively.
- LCM trained purely on text can handle v-Sonar visual embeddings in a zero-shot manner, achieving competitive performance with VLMs on video captioning tasks.
- OmniSONAR is easier to align than SONAR1 (embedding norm 1.69 vs. 0.264, covariance trace 1.83 vs. 0.049), indicating a representation collapse issue in the SONAR1 space.
- v-LCM matches state-of-the-art VLMs on image/video understanding in M3IT evaluation while significantly outperforming them in 61 non-English languages.

## Highlights & Insights

- A novel paradigm is proposed: unifying vision and language in a modality-agnostic continuous embedding space using a diffusion objective rather than discrete tokens.
- The success of the post-hoc alignment strategy demonstrates that a high-quality text embedding space can accommodate new modalities essentially "for free."
- The zero-shot visual understanding capability of LCM is impressive, validating the cross-modal transfer potential of shared embedding spaces.
- Multilingual capability is an inherent advantage: SONAR natively supports 1,500 languages, and v-LCM automatically inherits this coverage.

## Limitations & Future Work

- v-Sonar underperforms the original PE on Dream-1k retrieval (63.3 vs. 72.1), suggesting that alignment may sacrifice certain discriminative features.
- Performance on short-caption scenarios such as Vatex falls short of InternVL, likely due to training data bias toward detailed captions.
- The current v-LCM is relatively small-scale; direct comparison against large-scale VLMs (7B+) remains to be validated.
- The representation collapse issue in SONAR1 requires a more principled solution beyond relying on the OmniSONAR variant.

## Related Work & Insights

- This work contrasts with token-based multimodal models such as Chameleon, proposing continuous embedding spaces as a compelling alternative.
- The coarse-to-fine curriculum training strategy offers a transferable recipe for other cross-modal alignment tasks.
- The approach provides an important reference for developing multimodal models for low-resource languages.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The paradigm of aligning vision to a modality-agnostic embedding space combined with latent diffusion generation is highly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across retrieval, captioning, and multilingual benchmarks with complete ablations; large-scale comparisons are limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and fluent method description.
- **Value**: ⭐⭐⭐⭐⭐ — Opens a promising new direction for multimodal multilingual AI; the 61/62 language superiority is highly compelling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] UniHM: Unified Dexterous Hand Manipulation with Vision Language Model](unihm_unified_dexterous_hand_manipulation_with_vision_language_model.md)
- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[ICLR 2026\] Procedural Mistake Detection via Action Effect Modeling](procedural_mistake_detection_via_action_effect_modeling.md)
- [\[CVPR 2026\] DSCA: Dynamic Subspace Concept Alignment for Lifelong VLM Editing](../../CVPR2026/multimodal_vlm/dsca_dynamic_subspace_concept_alignment_for_lifelong_vlm_editing.md)
- [\[ICCV 2025\] MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding](../../ICCV2025/multimodal_vlm/musevl_modeling_unified_vlm_through_semantic_discrete_encodi.md)

</div>

<!-- RELATED:END -->
