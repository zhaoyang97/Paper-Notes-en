---
title: >-
  [Paper Note] Spatial Reasoning is Not a Free Lunch: A Controlled Study on LLaVA
description: >-
  [CVPR 2025][VLM Reasoning][Spatial Reasoning] By systematically replacing the image encoder (CLIP/SigLIP/SigLIP2/AIMv2) and introducing 2D-RoPE position embedding within the LLaVA framework, this study reveals that the spatial reasoning capability of VLMs is primarily determined by the encoder's training objective, and relying solely on 2D positional structures to improve spatial understanding is insufficient.
tags:
  - "CVPR 2025"
  - "VLM Reasoning"
  - "Spatial Reasoning"
  - "VLM"
  - "Image Encoder"
  - "2D Position Embedding"
  - "LLaVA"
date: 2026-05-08
content_hash: 707b9615b262d4dc
---

# Spatial Reasoning is Not a Free Lunch: A Controlled Study on LLaVA

**Conference**: CVPR 2025  
**arXiv**: [2603.12545](https://arxiv.org/abs/2603.12545)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Spatial Reasoning, VLM, Image Encoder, 2D Position Embedding, LLaVA

## TL;DR

By systematically replacing the image encoder (CLIP/SigLIP/SigLIP2/AIMv2) and introducing 2D-RoPE position embedding within the LLaVA framework, this study reveals that the spatial reasoning capability of VLMs is primarily determined by the encoder's training objective, and relying solely on 2D positional structures to improve spatial understanding is insufficient.

## Background & Motivation

**Background**: Currently, almost all VLMs rely on CLIP/SigLIP-like encoders trained via global image-text alignment. While these models perform exceptionally well on captioning and VQA, their spatial reasoning capabilities remain inconsistent.

**Limitations of Prior Work**: Even cutting-edge models such as Qwen2.5-VL and LLaVA-OneVision still exhibit fragile performance on spatial reasoning benchmarks (such as VSR, CountBenchQA, TopViewRS, etc.), with highly volatile scores.

**Key Challenge**: Existing VLM pipelines face two structural bottlenecks—(a) CLIP-like encoders optimize for global semantic alignment rather than fine-grained spatial details, and (b) images are flattened into 1D token sequences and processed with 1D position embeddings, discarding the 2D spatial structure.

**Goal**: Is the failure in spatial reasoning caused by the encoder's objective or the position embedding structure? What is the relative contribution of each factor?

**Key Insight**: Conducting controlled experiments within the LLaVA framework—fixing the language model and training data while strictly varying the choice of encoder and position embedding method to isolate the variables.

**Core Idea**: The bottleneck in spatial reasoning is not a data issue, but rather an architectural design limitation concerning the encoder's training objective and the positional encoding structure.

## Method

### Overall Architecture

Within the LLaVA-1.5 (7B) framework, the language model backbone and training data are fixed, while four distinct image encoders (CLIP, SigLIP, SigLIP2, AIMv2) are systematically evaluated. Each encoder is tested under standard 1D-RoPE and 2D-RoPE position embedding configurations, yielding eight variants in total. All variants undergo the same two-stage training pipeline (projection pretraining followed by instruction tuning), with images uniformly resized to $256 \times 256$.

### Key Designs

1. **Encoder Comparison**:

    - **CLIP**: The default encoder for the original LLaVA, trained via global image-text contrastive learning.
    - **SigLIP**: Replaces softmax with a sigmoid loss, improving negative sample treatment.
    - **SigLIP2**: An upgraded iteration of SigLIP.
    - **AIMv2**: Multimodal autoregressive pre-training, which simultaneously performs image-text alignment and autoregressive image feature prediction, yielding a denser training objective.
    - Core comparison dimension: Global contrastive objectives vs. dense/generative objectives.

2. **2D-RoPE Position Embedding**:

    - **Function**: Replaces standard 1D-RoPE with 2D-RoPE, encoding the horizontal and vertical indices of each patch.
    - **Mechanism**: Applied to the query and key projections in multimodal attention to preserve the 2D spatial structure of image patches.
    - **Design Motivation**: Traditional 1D-RoPE flattens 2D images into a 1D sequence, shedding spatial structure during the image-text fusion stage.

3. **Controlled Experimental Design**:

    - All variants share the same 7B LLaVA backbone.
    - Identical training data and hyperparameters are utilized.
    - Consequently, any performance discrepancies can be exclusively attributed to the choice of encoder or position embedding block.

### Loss & Training

Two-stage training: the first stage optimizes only the projection layer, while the second stage performs full-parameter instruction tuning. The original LLaVA dataset is utilized for training.

## Key Experimental Results

### Main Results

Comparing cutting-edge models and LLaVA variants across seven spatial reasoning benchmarks:

| Model | MMVP | CV-Bench 2D | TallyQA | GQA | VSR | TopViewRS | CountBenchQA |
|------|------|-------------|---------|-----|-----|-----------|--------------|
| Qwen2.5-VL (Frontier Best) | 0.770 | 0.754 | 60.4 | 89.1 | 0.456 | 0.891 | — |
| LLaVA v1.5 (CLIP baseline) | 0.577 | 0.490 | 33.2 | 55.8 | 0.384 | 0.468 | — |
| LLaVA-AIMv2 | 0.513 | **0.466** | **32.5** | 56.2 | 0.339 | **0.739** | — |
| LLaVA-AIMv2 + 2D-RoPE | **0.560** | 0.432 | 32.3 | **60.3** | **0.338** | 0.719 | — |
| LLaVA-SigLIP | 0.433 | 0.412 | 25.6 | 54.9 | 0.349 | 0.581 | — |
| LLaVA-SigLIP2 | 0.427 | 0.442 | 24.0 | 52.7 | 0.371 | 0.532 | — |

### Ablation Study (Effects of 2D-RoPE)

| Encoder | Standard 1D-RoPE → +2D-RoPE | MMVP | CV-Bench 2D | GQA | VSR |
|--------|------------------------|------|-------------|-----|-----|
| CLIP | 1D → 2D | 0.577→0.513 ↓ | 0.490→0.443 ↓ | 55.8→57.2 ↑ | 0.384→0.283 ↓ |
| SigLIP | 1D → 2D | 0.433→0.507 ↑ | 0.412→0.425 ↑ | 54.9→57.7 ↑ | 0.349→0.295 ↓ |
| AIMv2 | 1D → 2D | 0.513→0.560 ↑ | 0.466→0.432 ↓ | 56.2→60.3 ↑ | 0.339→0.338 — |

### Key Findings

- **AIMv2 encoder outperforms the CLIP baseline on most spatial benchmarks**, demonstrating significant performance gains on TopViewRS (+27%) and GQA (+4.4%), which validates that dense/generative training objectives help retain spatial details.
- **Unstable performance of 2D-RoPE**: While it benefits AIMv2 on MMVP and GQA, it leads to a severe performance drop (0.384 → 0.283) for the CLIP baseline on VSR, indicating that the impact of position embedding depends heavily on the quality of characteristics provided by the encoder.
- **Huge gap between frontier models and LLaVA variants**: Qwen2.5-VL significantly outperforms all LLaVA variants across almost all benchmarks, showing that encoder replacement is only a partial solution; model scale, training data, and tight architectural integration are also crucial.
- **SigLIP family lags behind the CLIP baseline on spatial tasks**: Surprisingly, a supposedly superior training objective does not necessarily yield better spatial reasoning. This might stem from a trade-off between SigLIP's global semantic optimization and spatial detail capture.

## Highlights & Insights

- **Exceptionally clear controlled experimental design**: By keeping all other variables constant and only varying the encoder and positional encoding, the study ensures robust conclusions. This diagnostic methodology serves as an excellent reference for isolating individual variables in complex systems.
- **Cross-task advantages of AIMv2**: AIMv2 employs an autoregressive prediction target, essentially forcing the encoder to model relationships between local patches, which naturally benefits spatial reasoning. This implies that "how the encoder is trained" is more critical than "which position embedding is used."
- **Insightful failure cases of 2D-RoPE**: Merely introducing 2D spatial coordinates may conflict with the LLM's native 1D-RoPE, leading to confusion during multimodal attention computation. This suggests that positional encoding design for the multimodal fusion stage requires more holistic architectural considerations.

## Limitations & Future Work

- **Resolution Constraints**: All experiments were fixed at $256 \times 256$. Since modern VLMs leverage much higher resolutions (448, 672, or even dynamic resolutions), this low resolution inherently limits spatial reasoning performance.
- **Omission of DINOv2**: As a representative self-supervised encoder, DINOv2 prioritizes local features and might be highly advantageous for spatial reasoning, yet it was not included in this study.
- **Data Scale Discrepancies**: Frontier models are trained on datasets that are orders of magnitude larger, making the comparison with the baseline LLaVA variants somewhat unfair.
- **Focus Placed Solely on 2D Spatial Reasoning**: More complex spatial tasks such as 3D physical scene understanding and depth estimation were not explored.
- **Lack of In-depth Analysis on standard positional conflicts**: The mismatch between 2D-RoPE and native LLM position embeddings is only reported empirically, lacking a rigorous theoretical explanation for why 2D-RoPE sometimes degrades performance.

## Related Work & Insights

- **vs. Qwen2-VL**: Qwen2-VL introduces multimodal RoPE to preserve aspect ratio information, achieving the best performance on spatial tasks. The controlled experiments in this study provide theoretical validation for this design choice while cautioning that positional encoding is not a panacea.
- **vs. SpatialVLM / SpatialRGPT**: These existing works enhance spatial reasoning via data augmentation and dedicated architectural modules, whereas this study demonstrates that the pre-training objectives of the image encoder are the underlying driving factor.
- This paper provides clear design guidelines for "how to construct spatial-aware VLMs": priority should be placed on selecting encoders with dense/generative training objectives, while positional encoding should be regarded as a secondary auxiliary component.

## Rating

- Novelty: ⭐⭐⭐ While the methodology is not highly novel (mainly replacing encoders and adding 2D-RoPE), the diagnostic perspective is of high value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across 7 benchmarks and 8 variants; however, fixing the resolution at 256 and omitting DINOv2 are slight drawbacks.
- Writing Quality: ⭐⭐⭐⭐ The writing is concise, clear, and the experimental design is thoroughly described.
- Value: ⭐⭐⭐⭐ Provides definitive guidance to the VLM spatial reasoning community, highlighting that upgrading the encoder, rather than merely tweaking position embeddings, is key.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] To Think or Not To Think: A Study of Explicit Thinking in Rule-Based Visual Reinforcement Fine-Tuning](../../NeurIPS2025/vlm_reasoning/think_or_not_think_a_study_of_explicit_thinking_in_rule-based_visual_reinforceme.md)
- [\[NeurIPS 2025\] Can LLMs Reason Over Non-Text Modalities in a Training-Free Manner? A Case Study with In-Context Representation Learning](../../NeurIPS2025/vlm_reasoning/can_llms_reason_over_non-text_modalities_in_a_training-free_manner_a_case_study_.md)
- [\[CVPR 2025\] ESPIRE: A Diagnostic Benchmark for Embodied Spatial Reasoning of Vision-Language Models](espire_a_diagnostic_benchmark_for_embodied_spatial_reasoning_of_vision-language_.md)
- [\[ICCV 2025\] Training-Free Personalization via Retrieval and Reasoning on Fingerprints](../../ICCV2025/vlm_reasoning/training-free_personalization_via_retrieval_and_reasoning_on_fingerprints.md)
- [\[ICCV 2025\] LLaVA-CoT: Let Vision Language Models Reason Step-by-Step](../../ICCV2025/vlm_reasoning/llava-cot_let_vision_language_models_reason_step-by-step.md)

</div>

<!-- RELATED:END -->
