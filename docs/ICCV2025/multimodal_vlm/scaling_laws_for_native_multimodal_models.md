---
title: >-
  [Paper Note] Scaling Laws for Native Multimodal Models
description: >-
  [ICCV 2025 (Oral)][Multimodal VLM][native multimodal] By training 457 models across diverse architectures, scales, and training data mixtures, this paper systematically investigates scaling laws for Native Multimodal Models (NMMs). It finds that early-fusion architectures (without pretrained visual encoders) outperform late-fusion counterparts at small parameter scales, are more training-efficient, and simpler to deploy; incorporating MoE further yields substantial performance gains.
tags:
  - ICCV 2025 (Oral)
  - Multimodal VLM
  - native multimodal
  - early fusion
  - late fusion
  - scaling laws
  - MoE
date: 2026-05-08
content_hash: 671fcb51ea10bddd
---

# Scaling Laws for Native Multimodal Models

**Conference**: ICCV 2025 (Oral)
**arXiv**: [2504.07951](https://arxiv.org/abs/2504.07951)
**Code**: N/A
**Area**: Multimodal VLM / Scaling Laws
**Keywords**: native multimodal, early fusion, late fusion, scaling laws, MoE

## TL;DR

By training 457 models across diverse architectures, scales, and training data mixtures, this paper systematically investigates scaling laws for Native Multimodal Models (NMMs). It finds that early-fusion architectures (without pretrained visual encoders) outperform late-fusion counterparts at small parameter scales, are more training-efficient, and simpler to deploy; incorporating MoE further yields substantial performance gains.

## Background & Motivation

**State of the Field**: Mainstream VLMs (e.g., LLaVA, InternVL) adopt late-fusion architectures — a visual encoder (e.g., CLIP-ViT) and an LLM are pretrained independently and then connected via a connector for multimodal training. This paradigm offers high sample efficiency, but whether it holds an inherent architectural advantage remains an open question.

**Limitations of Prior Work**: (1) The visual encoder in late-fusion architectures imposes fixed-resolution and aspect-ratio constraints, and coordinating multiple components increases engineering complexity. (2) Native Multimodal Models (NMMs), trained from scratch on all modalities jointly, lack systematic architectural comparisons and scaling law analyses. (3) The community has assumed late-fusion to be superior without sufficient empirical evidence.

**Root Cause**: Under a fixed compute budget, which architecture — early-fusion or late-fusion — should NMMs adopt? How does scaling behavior differ across model sizes and data volumes?

**Starting Point**: A large-scale empirical study training 457 models covering diverse combinations of architecture, scale, data mixture, and MoE configuration, with quantitative conclusions derived by fitting scaling laws.

## Method

### Overall Architecture

This is a systematic empirical study. The authors cover the following dimensions:

- **Architecture**: (a) Early-fusion: no visual encoder; raw image patches are fed directly into a unified Transformer; (b) Late-fusion: pretrained visual encoder (e.g., CLIP-ViT) + connector + LLM; (c) Visual tokenizer: images are first discretized into token sequences before being fed into the model.
- **Model scale**: Multiple parameter configurations ranging from small to large.
- **Training data**: Varying image-text data mixture ratios.
- **MoE configuration**: Different numbers of experts and activation ratios.

Standard power-law fitting is applied to relate validation loss to model parameter count and training token count under each configuration.

### Key Designs

1. **Systematic Comparison of Early-Fusion vs. Late-Fusion**

    - Core finding: Under identical parameter counts and training data, early-fusion is no worse than late-fusion — directly challenging the community consensus that "CLIP + LLM is the optimal paradigm."
    - Further finding: At smaller parameter scales, early-fusion is actually superior, as it does not need to allocate parameters and compute to a separate visual encoder.
    - Advantages of early-fusion: (a) More training-efficient — no need to pretrain visual components independently; (b) Simpler deployment — a single unified model; (c) Greater flexibility — not constrained by the visual encoder's resolution or aspect ratio.

2. **MoE for NMMs**

    - Mixture of Experts is introduced into NMMs, allowing the model to learn modality-specific weight paths.
    - **Design Motivation**: Inter-modality interference is a central challenge for NMMs — training signals from vision and text may conflict, and MoE provides efficient implicit decoupling.
    - MoE yields particularly significant gains on early-fusion architectures, further confirming the importance of modality decoupling.

3. **Disadvantages of Visual Tokenizers**

    - Discrete visual token schemes perform worst across all settings — information is irrecoverably lost during quantization.
    - This provides scaling-law-level empirical evidence for the debate on continuous vs. discrete visual representations.

### Loss & Training

Standard next-token prediction is used for text; visual loss configurations vary across architectural variants. Scaling laws follow the standard power-law form: $L(N,D) = aN^{-\alpha} + bD^{-\beta} + c$, where $N$ denotes parameter count and $D$ denotes the number of training tokens.

## Key Experimental Results

### Main Results

| Finding | Details |
|---------|---------|
| Total models trained | 457, covering diverse architecture × scale × data mixture × MoE configurations |
| Early-fusion vs. Late-fusion | Small scale: early-fusion outperforms late-fusion; large scale: comparable performance |
| Early-fusion efficiency | Requires fewer training FLOPs to reach the same validation loss |
| MoE gains | Consistently yields significant performance improvements across all architectural variants |
| Visual tokenizer | Underperforms continuous representation schemes at all scales |

### Ablation Study

| Factor | Key Observation |
|--------|----------------|
| Data mixture ratio | Early-fusion is more sensitive to the proportion of visual data and requires more |
| Number of MoE experts | An optimal range exists; too many experts degrades performance at small scales |
| Model scale | Early-fusion's advantage diminishes but does not reverse as scale increases |
| Scaling law extrapolation | Small-scale experiments reliably predict large-scale training outcomes |

### Key Findings

- **Central conclusion**: Late-fusion architectures hold no inherent advantage — early-fusion performs comparably or better under matched settings.
- Scaling laws fitted on small models extrapolate accurately to large models, reducing the trial-and-error cost of NMM research.
- MoE is a key component for NMMs — modality-specific routing effectively alleviates inter-modality interference.
- An exceptionally detailed analysis comprising 28 figures and 13 tables provides a comprehensive empirical foundation for NMM architecture selection.

## Highlights & Insights

- **ICCV Oral; a systematic study at the scale of 457 models** is unprecedented and establishes a scientific foundation for the NMM field.
- The finding that "a pretrained visual encoder is not necessary" is a paradigm-level contribution, forming a coherent narrative with findings from EVEv2 and Web-SSL.
- Scaling laws move NMM research from "trial-and-error" to "prediction" — small-model experiments can forecast large-model behavior, substantially reducing costs.
- Authored at Apple (Joshua Susskind), reflecting strong industrial interest in the NMM direction.

## Limitations & Future Work

- Despite 457 models, the largest scale remains compute-constrained — the reliability of extrapolation beyond 100B+ parameters has not been validated.
- Scaling laws have not yet been verified on text-to-image or video generation tasks.
- The impact of data quality remains underexplored — high-quality annotations may shift the relative advantage between early- and late-fusion.
- Neither open-source models nor training code are provided, limiting reproducibility.

## Related Work & Insights

- **vs. EVEv2**: EVEv2 focuses on optimal training strategies (Divide-and-Conquer) for encoder-free VLMs; this paper provides more systematic architectural comparisons and scaling laws — the two are highly complementary.
- **vs. Chinchilla/Kaplan scaling laws**: This work extends the LLM scaling law methodology to the multimodal setting, filling a critical gap for NMMs.
- **vs. Mono-InternVL**: Mono-InternVL represents engineering practice for encoder-free VLMs; this paper is a systematic scientific study.
- **Implication**: If early-fusion NMMs are sufficiently capable, the default paradigm of the VLM community (CLIP + LLM) may warrant fundamental reconsideration.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Large-scale empirical study at 457 models is unprecedented; the finding that early-fusion matches late-fusion is a paradigm-level contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 28 figures and 13 tables covering comprehensive analysis across architecture, scale, data, and MoE.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Oral-quality scientific narrative with clear and compelling conclusions.
- **Value**: ⭐⭐⭐⭐⭐ Profound guidance for architecture selection in the VLM community; scaling laws establish a scientific foundation for NMM research.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints](../../NeurIPS2025/multimodal_vlm/navil_rethinking_scaling_properties_of_native_multimodal_large_language_models_u.md)
- [\[CVPR 2026\] Scaling Spatial Intelligence with Multimodal Foundation Models](../../CVPR2026/multimodal_vlm/scaling_spatial_intelligence_with_multimodal_foundation_models.md)
- [\[ICCV 2025\] Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy](dita_scaling_diffusion_transformer_for_generalist_visionlang.md)
- [\[ICCV 2025\] Scaling Inference-Time Search with Vision Value Model for Improved Visual Comprehension](scaling_inferencetime_search_with_vision_value_model_for_imp.md)
- [\[NeurIPS 2025\] SITCOM: Scaling Inference-Time COMpute for VLAs](../../NeurIPS2025/multimodal_vlm/sitcom_scaling_inference-time_compute_for_vlas.md)

<!-- RELATED:END -->
