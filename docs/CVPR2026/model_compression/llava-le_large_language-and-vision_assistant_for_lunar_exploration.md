---
title: >-
  [Paper Note] LLaVA-LE: Large Language-and-Vision Assistant for Lunar Exploration
description: >-
  [CVPR 2026][Model Compression][Lunar exploration] LLaVA-LE is the first vision-language model tailored for lunar exploration. By constructing LUCID, a large-scale real lunar image-text dataset (96K images + 81K QA pairs), and applying two-stage curriculum fine-tuning on LLaVA, the model achieves a 3.3× improvement over the baseline on lunar geological understanding and multimodal reasoning.
tags:
  - CVPR 2026
  - Model Compression
  - Lunar exploration
  - vision-language model
  - geological understanding
  - multimodal reasoning
  - domain fine-tuning
date: 2026-05-08
content_hash: b07bf624e109e859
---

# LLaVA-LE: Large Language-and-Vision Assistant for Lunar Exploration

**Conference**: CVPR 2026
**arXiv**: [2603.24696](https://arxiv.org/abs/2603.24696)
**Code**: [https://github.com/OSUPCVLab/LLaVA-LE](https://github.com/OSUPCVLab/LLaVA-LE)
**Area**: Model Compression
**Keywords**: Lunar exploration, vision-language model, geological understanding, multimodal reasoning, domain fine-tuning

## TL;DR

LLaVA-LE is the first vision-language model tailored for lunar exploration. By constructing LUCID, a large-scale real lunar image-text dataset (96K images + 81K QA pairs), and applying two-stage curriculum fine-tuning on LLaVA, the model achieves a 3.3× improvement over the baseline on lunar geological understanding and multimodal reasoning.

## Background & Motivation

VLMs have made remarkable progress in natural image understanding, yet remain nearly absent in planetary science. The primary obstacle is the lack of large-scale, high-quality paired planetary image-text data. Existing lunar datasets are small, unimodal, and often contain synthetic data, making them unsuitable for training modern VLMs.

**Key Challenge**: Planetary remote sensing is fundamentally different from natural image understanding — lunar geological analysis requires joint reasoning across physical modalities (optical, gravity anomaly, topographic slope), whereas a single image provides only surface reflectance information, which is insufficient for understanding geological structure.

**Goal**: To construct the first large-scale multimodal lunar dataset grounded in real NASA mission data, and to train a vision-language assistant capable of lunar geological description, geological question answering, and multimodal reasoning.

## Method

### Overall Architecture

Data construction → two-stage fine-tuning. Data are sourced from three NASA missions: LROC (high-resolution optical), GRAIL (gravity anomaly), and LOLA (topographic slope). Scientific descriptions and QA pairs are generated via GPT-5. The model is built on the LLaVA framework, employing a CLIP visual encoder paired with an LLM and trained in two stages.

### Key Designs

1. **LUCID Dataset Construction**:

    - **Function**: Provides 96K panchromatic images with detailed scientific descriptions and 81K VQA pairs.
    - **Mechanism**: Panchromatic lunar images are sourced from LROC WAC; structured prompts are used to invoke GPT-5 to generate detailed scientific descriptions covering geological context, topographic morphology, and inferred subsurface features. Three categories of QA pairs are then derived from these descriptions: detailed description, conversation, and reasoning.
    - **Design Motivation**: Combining real data with GPT-5 annotation balances dataset scale with annotation quality.

2. **Two-Stage Curriculum Learning**:

    - **Function**: Progressively adapts a general-purpose VLM to the planetary science domain.
    - **Mechanism**: Stage 1 (concept alignment) — fine-tunes on image-description pairs to teach the model domain-specific lunar geological terminology and visual-semantic mappings. Stage 2 (instruction tuning) — fine-tunes on QA pairs to enhance interactive question answering and reasoning capabilities.
    - **Design Motivation**: Direct instruction tuning without prior concept alignment yields suboptimal results; establishing a domain conceptual foundation first is necessary.

3. **Multi-Level Evaluation Benchmark**:

    - **Function**: Assesses model performance across varying levels of reasoning complexity.
    - **Mechanism**: Three evaluation dimensions are designed — Detailed (descriptive), Conversation (dialogic), and Reasoning — scored by a dual-judge system using GPT-4 and Gemini.
    - **Design Motivation**: A single metric is insufficient to evaluate a domain-specific VLM; multidimensional measurement is required.

### Loss & Training

Standard LLaVA training strategy: Stage 1 freezes the LLM and trains only the projection layer for alignment; Stage 2 unfreezes all parameters for full instruction tuning.

## Key Experimental Results

### Main Results

| Model | Detailed | Conversation | Reasoning | Overall | Relative Judge Score |
|-------|----------|--------------|-----------|---------|----------------------|
| Base LLaVA | Low | Low | Low | ~0.32 | — |
| LLaVA-LE Stage 1 | Medium | Medium | Medium | ~0.51 | — |
| LLaVA-LE Stage 2 | High | High | 1.070 | ~1.06 | Exceeds reference |

LLaVA-LE Stage 2 achieves a 3.3× overall improvement over Base LLaVA. The reasoning dimension score of 1.070 surpasses the judge's own reference answers.

### Ablation Study

| Configuration | Overall | Notes |
|---------------|---------|-------|
| Base LLaVA (no fine-tuning) | ~0.32 | General-purpose model performs poorly on lunar domain |
| Stage 1 only | ~0.51 | Concept alignment contributes ~60% improvement |
| Stage 1 + Stage 2 | ~1.06 | Instruction tuning approximately doubles the gain |

### Key Findings

- General-purpose VLMs are nearly unusable in planetary science; domain fine-tuning is critical.
- The concept alignment in Stage 1 contributes substantially, demonstrating that establishing domain terminology and conceptual mappings is foundational.
- Reasoning scores exceeding the judge's reference answers indicate that data-intensive training enables the model to produce high-quality geological analyses.

## Highlights & Insights

- **First planetary science VLM**: Fills a gap in AI for planetary exploration and establishes a new application direction.
- **GPT-5-based scientific annotation pipeline**: The approach of using large models to automatically generate high-quality annotations for domain-specific data is transferable to other scientific domains with limited labeled resources.
- **Fully open-source**: The dataset, code, and model weights are all publicly released, offering significant value to follow-up research.

## Limitations & Future Work

- The current work uses only panchromatic images, leaving the joint reasoning potential of multimodal remote sensing data (gravity, slope) largely unexploited.
- GPT-5-generated annotations may contain geological inaccuracies and require expert validation.
- Evaluation still relies on LLM judges, lacking human assessment by planetary scientists.
- Future work may extend the framework to other celestial bodies such as Mars and asteroids.

## Related Work & Insights

- **vs. LLaVA-Med**: LLaVA-Med adapts LLaVA to medicine; LLaVA-LE adapts it to planetary science. The approaches are conceptually similar, but the domain challenges differ substantially.
- **vs. Space-LLaVA**: Space-LLaVA relies on synthetic data, whereas LLaVA-LE uses real NASA data, yielding higher data quality.
- **vs. AlphaEarth**: AlphaEarth targets Earth observation; LLaVA-LE targets the Moon, where data scarcity presents a greater challenge.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Innovative domain application, though the methodology (LLaVA fine-tuning) is relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐ — Evaluation design is reasonable but limited in scale; comparisons with more baselines are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated; dataset construction is described in detail.
- **Value**: ⭐⭐⭐⭐ — The open-source dataset and model are of significant importance to the planetary science community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[NeurIPS 2025\] Vision-centric Token Compression in Large Language Model](../../NeurIPS2025/model_compression/vision-centric_token_compression_in_large_language_model.md)
- [\[ICCV 2025\] B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens](../../ICCV2025/model_compression/b_vllm_a_vision_large_language_model_with_balanced_spatio_temporal_tokens.md)
- [\[ICLR 2026\] AMiD: Knowledge Distillation for LLMs with α-mixture Assistant Distribution](../../ICLR2026/model_compression/amid_knowledge_distillation_for_llms_with_α-mixture_assistant_distribution.md)
- [\[NeurIPS 2025\] Ensemble++: Scalable Exploration via Ensemble](../../NeurIPS2025/model_compression/scalable_exploration_via_ensemble.md)

</div>

<!-- RELATED:END -->
