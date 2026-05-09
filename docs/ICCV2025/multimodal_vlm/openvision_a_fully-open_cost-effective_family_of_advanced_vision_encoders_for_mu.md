---
title: >-
  [Paper Note] OpenVision: A Fully-Open, Cost-Effective Family of Advanced Vision Encoders for Multimodal Learning
description: >-
  [ICCV 2025][Multimodal VLM][CLIP] This paper introduces OpenVision — a fully open-source (data, training code, and weights) family of vision encoders (5.9M–632.1M parameters) trained on the CLIPS framework with the Recap-DataComp-1B dataset. When integrated into multimodal frameworks such as LLaVA, OpenVision matches or surpasses OpenAI CLIP and Google SigLIP, providing the community with a transparent and flexible alternative visual backbone.
tags:
  - ICCV 2025
  - Multimodal VLM
  - CLIP
  - Vision Encoder
  - Open Source
  - Multimodal Learning
  - LLaVA
date: 2026-05-08
content_hash: 60fba58a24f7a158
---

# OpenVision: A Fully-Open, Cost-Effective Family of Advanced Vision Encoders for Multimodal Learning

**Conference**: ICCV 2025
**arXiv**: [2505.04601](https://arxiv.org/abs/2505.04601)
**Code**: [https://github.com/UCSC-VLAA/OpenVision](https://github.com/UCSC-VLAA/OpenVision)
**Area**: Multimodal VLM / Vision Encoders
**Keywords**: CLIP, Vision Encoder, Open Source, Multimodal Learning, LLaVA

## TL;DR

This paper introduces OpenVision — a fully open-source (data, training code, and weights) family of vision encoders (5.9M–632.1M parameters) trained on the CLIPS framework with the Recap-DataComp-1B dataset. When integrated into multimodal frameworks such as LLaVA, OpenVision matches or surpasses OpenAI CLIP and Google SigLIP, providing the community with a transparent and flexible alternative visual backbone.

## Background & Motivation

**Background**: Nearly all multimodal foundation models (LLaVA, Mini-GPT-4, Falcon2 VLM, Eagle, etc.) rely on OpenAI's CLIP-L/336 as the vision encoder — a de facto standard that has dominated the multimodal landscape.

**Limitations of Prior Work**:
   - **Opacity**: OpenAI CLIP's training data and full training recipe are not publicly available, limiting reproducibility and transparency.
   - **Limited Scale**: Only Base and Large variants exist, leaving a gap for lightweight edge deployment and exploration of larger models.
   - **Known Deficiencies**: CLIP exhibits well-documented hallucination problems in spatial relation understanding and object counting.
   - **Insufficient Open-Source Alternatives**: Existing open-source CLIP models (OpenCLIP, DataComp, DFN, etc.) achieve strong zero-shot performance but fall noticeably short on multimodal downstream tasks (MME, ChartQA, TextVQA).

**Key Challenge**: The community requires a fully open visual backbone to advance multimodal research, yet no from-scratch open-source encoder has consistently matched or exceeded OpenAI CLIP on multimodal benchmarks.

**Goal**: To provide a fully open (data + code + weights), cost-effective, and multi-scale family of vision encoders that reaches or exceeds closed-source performance in multimodal learning scenarios.

**Key Insight**: Building on recent open-source advances — the CLIPS training framework (multi-positive contrastive learning with a text generator) and Recap-DataComp-1B (a recaptioned billion-scale dataset) — the paper systematically analyzes critical design choices.

**Core Idea**: Standing on the shoulders of CLIPS and Recap-DataComp-1B, the paper systematically tunes training strategies and releases 25+ fully open-source vision encoder checkpoints that comprehensively match or surpass OpenAI CLIP.

## Method

### Overall Architecture

OpenVision follows the standard dual-tower CLIP architecture: a vision encoder and a text encoder trained with contrastive loss. Key enhancements come from the CLIPS framework, which introduces a multi-positive loss (treating both original and synthetic captions as positives) and jointly trains a lightweight text decoder to generate new captions. After training, only the visual backbone is retained; the text tower and decoder are discarded.

### Key Designs

1. **Multi-Stage Resolution Training**

    - **Function**: Gradually trains from lower to higher resolutions in successive stages to improve training efficiency.
    - **Mechanism**: For Large/SoViT-400M/Huge variants, training proceeds through $84 \times 84$, $224 \times 224$, and $336 \times 336$ (or $384 \times 384$) stages sequentially; smaller Tiny/Small/Base models begin at $160 \times 160$. The three stages process 12.8B, 1.024B, and 256M image-text pairs with global batch sizes of 32K, 16K, and 8K, respectively.
    - **Design Motivation**: Following the efficient training curriculum of CLIPA, the low-resolution stage rapidly learns semantic representations while the high-resolution stage refines fine-grained details, substantially reducing computational cost.

2. **CLIPS Multi-Positive Contrastive Learning**

    - **Function**: Extends standard CLIP contrastive learning by treating both original and synthetic captions as positive pairs.
    - **Mechanism**: CLIPS leverages Recap-DataComp-1B (DataComp-1B recaptioned with LLaVA) to supply richer synthetic captions, and jointly trains a text decoder to generate additional captions, further enriching the training signal.
    - **Design Motivation**: A single caption may be insufficient or noisy; the multi-positive strategy increases training signal diversity, and synthetic captions are generally more detailed and accurate.

3. **Flexible Model Scale and Patch Size**

    - **Function**: Releases a complete encoder series from Ti (5.9M) to H (632.1M) parameters and explores different patch sizes (8×8 vs. 16×16).
    - **Mechanism**: Fixed sinusoidal positional encodings allow adaptation to varying sequence lengths. Smaller patches provide finer spatial resolution at higher computational cost.
    - **Design Motivation**: Addresses diverse deployment requirements ranging from edge devices to servers. An 8×8 patch yields substantial gains on TextVQA (Tiny +4.4%, Small +5.0%, Base +3.3%) at the cost of a significantly larger number of visual tokens.

4. **Role of Synthetic Captions and Auxiliary Decoder**

    - **Function**: Ablation analysis of the contributions of synthetic captions and the auxiliary text decoder to encoder quality.
    - **Mechanism**: Experiments show that both components positively contribute to multimodal downstream tasks. Removing synthetic captions degrades performance on most benchmarks; removing the auxiliary decoder also results in a clear negative impact.
    - **Design Motivation**: Validates the specific contribution of each component in the CLIPS framework to multimodal scenarios.

### Loss & Training

- Contrastive loss + multi-positive loss + text generation loss (from CLIPS)
- Three-stage learning rate schedule with cosine decay; base learning rates are $8 \times 10^{-6}$, $4 \times 10^{-7}$, and $1 \times 10^{-7}$, respectively
- Text encoder input: 80 tokens; text decoder generation: 128 tokens
- Downstream evaluation conducted under two frameworks: LLaVA-1.5 (frozen encoder) and Open-LLaVA-Next (full fine-tuning)

## Key Experimental Results

### Main Results: Multimodal Performance Comparison under LLaVA-1.5

| Vision Encoder | Resolution | TextVQA | ChartQA | OCR | MME | SEED | GQA | POPE |
|---|---|---|---|---|---|---|---|---|
| OpenAI-CLIP L/14 | 224 | 56.1 | 13.2 | 177 | 1443/306 | 66.0 | 60.8 | 85.0 |
| LAION-2B-CLIP L/14 | 224 | 54.2 | 12.8 | 165 | 1434/298 | 65.5 | 59.0 | 84.5 |
| DataComp-1B-CLIP L/14 | 224 | 53.0 | 12.3 | 131 | 1382/312 | 62.4 | 57.8 | 83.0 |
| DFN-2B-CLIP L/14 | 224 | 53.2 | 12.4 | 246 | 1447/306 | 65.6 | 59.1 | 85.0 |
| **OpenVision L/14** | **224** | **57.7** | **13.9** | **315** | **1487/317** | **69.5** | **62.9** | **86.4** |
| OpenAI-CLIP L/14 | 336 | 59.1 | 13.8 | 201 | 1475/288 | 67.5 | 61.1 | 85.7 |
| **OpenVision L/14** | **336** | **61.2** | **15.7** | **339** | **1525/315** | **70.5** | **63.7** | **87.2** |
| SigLIP SoViT-400M/14 | 384 | 62.6 | 14.5 | 338 | 1481/347 | 69.4 | 63.3 | 87.0 |
| **OpenVision SoViT-400M/14** | **384** | **62.4** | **16.1** | **357** | **1493/320** | **70.4** | **63.8** | **88.0** |

### Ablation Study: Effect of Different Patch Sizes (LLaVA-1.5)

| Encoder | Patch | TextVQA | ChartQA | OCR | MME | SEED | GQA |
|---|---|---|---|---|---|---|---|
| Ti/16 | 16 | 50.2 | 11.6 | 139 | 1329/280 | 62.0 | 58.0 |
| Ti/8 | 8 | **54.6** | **12.9** | **223** | **1383/310** | **66.3** | **59.7** |
| S/16 | 16 | 54.3 | 12.0 | 235 | 1393/343 | 67.5 | 61.6 |
| S/8 | 8 | **59.3** | **15.9** | **310** | **1449/303** | **70.3** | **62.0** |
| B/16 | 16 | 57.9 | 14.5 | 293 | 1432/333 | 69.8 | 62.8 |
| B/8 | 8 | **61.2** | **17.2** | **345** | **1545/299** | **71.8** | **63.0** |

### Key Findings
- **OpenVision comprehensively outperforms open-source CLIP alternatives**: Under equivalent settings, LAION-2B, DataComp-1B, and DFN-2B CLIP models all fall notably short of OpenVision on multimodal tasks, despite potentially higher zero-shot classification performance (e.g., DFN-2B achieves 81.4% on ImageNet vs. OpenVision's 78.4%). This demonstrates that **zero-shot classification performance is not a sufficient indicator of multimodal quality**.
- **Matches/surpasses closed-source models**: OpenVision L/14@336 outperforms OpenAI CLIP L/14@336 on most LLaVA-1.5 benchmarks and achieves comparable results under Open-LLaVA-Next.
- **Smaller patches substantially improve fine-grained understanding**: 8×8 patches yield +3–5% gains on TextVQA and OCR, at the cost of a fourfold increase in visual token count.
- **Extremely small models remain competitive**: The ~250M model composed of Ti/16 (5.9M) + Qwen2.5-0.5B, while showing reduced performance, retains 87% of the average performance.
- **Scaling training benefits downstream tasks**: H/14 (632.1M parameters) achieves further improvements on multiple metrics, confirming the value of larger encoders for multimodal applications.

## Highlights & Insights

- **Value of full transparency**: When data, code, and weights are all publicly released, researchers can conduct systematic ablations to understand what factors truly matter. This work fills the gap of a fully qualifying open-source vision encoder for multimodal scenarios.
- **CLIP zero-shot performance ≠ multimodal performance**: This is an important finding — DFN-2B CLIP achieves 81.4% on ImageNet yet lags clearly behind OpenVision (78.4%) on multimodal tasks. Training strategies (e.g., synthetic captions, multi-positive loss) have a greater impact on multimodal downstream performance than classification accuracy.
- **High practical value**: 25+ checkpoints spanning a complete scale spectrum from 5.9M to 632.1M, across multiple patch sizes and resolutions, provide the community with a comprehensive "buffet" of multimodal visual backbones.

## Limitations & Future Work

- The training data (Recap-DataComp-1B), while open-source, has recaptioning quality bounded by the capabilities of the LLaVA model used for annotation.
- More advanced architectural variants (e.g., the architectural improvements in EVA-CLIP) are not explored.
- Evaluation on temporal multimodal tasks such as video understanding is absent.
- Comparison with newer models such as SigLIP2 is insufficiently thorough.
- Multimodal evaluation is primarily conducted under the LLaVA framework; performance under other frameworks such as Qwen-VL and InternVL remains unknown.

## Related Work & Insights

- **vs. OpenAI CLIP**: OpenAI CLIP's training data is not publicly available and only B/L scales are provided; OpenVision offers a fully transparent training pipeline and a richer range of model scales, matching or exceeding it on multimodal downstream tasks.
- **vs. SigLIP**: Google SigLIP's training data is likewise undisclosed; OpenVision SoViT-400M/14 achieves comparable or superior performance at the same parameter scale.
- **vs. OpenCLIP/LAION/DataComp/DFN**: These open-source CLIP models perform well on zero-shot classification but fall notably short on multimodal downstream tasks; OpenVision bridges this gap through the CLIPS framework and synthetic captions.

## Rating

- Novelty: ⭐⭐⭐ — Primarily a systematic integration and engineering optimization of existing techniques (CLIPS + DataComp); methodological innovation is limited.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive comparisons and ablations across multiple frameworks (LLaVA-1.5/Open-LLaVA-Next), scales, resolutions, and patch sizes.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, information-dense tables, and well-articulated conclusions.
- Value: ⭐⭐⭐⭐⭐ — Significant community contribution; 25+ open-source checkpoints provide the much-needed transparent and flexible visual backbone for multimodal research.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Dynamic Multimodal Prototype Learning in Vision-Language Models](dynamic_multimodal_prototype_learning_in_vision-language_models.md)
- [\[ICCV 2025\] On Large Multimodal Models as Open-World Image Classifiers](on_large_multimodal_models_as_open-world_image_classifiers.md)
- [\[ICCV 2025\] Effective Training Data Synthesis for Improving MLLM Chart Understanding](effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)
- [\[ICCV 2025\] HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?](hrscene_how_far_are_vlms_from_effective_high-resolution_image_understanding.md)
- [\[ICCV 2025\] LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)

<!-- RELATED:END -->
