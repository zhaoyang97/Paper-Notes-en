---
title: >-
  [Paper Note] PixDLM: A Dual-Path Multimodal Language Model for UAV Reasoning Segmentation
description: >-
  [CVPR 2026][Segmentation][UAV reasoning segmentation] This paper formally defines the UAV Reasoning Segmentation task, constructs the DRSeg benchmark comprising 10K high-resolution UAV images with chain-of-thought reason…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "UAV reasoning segmentation"
  - "multimodal large language model"
  - "dual-path visual encoder"
  - "chain-of-thought reasoning"
  - "pixel-level prediction"
date: 2026-05-08
content_hash: 6d2326a3291794d6
---

# PixDLM: A Dual-Path Multimodal Language Model for UAV Reasoning Segmentation

**Conference**: CVPR 2026
**arXiv**: [2604.15670](https://arxiv.org/abs/2604.15670)
**Code**: [https://github.com/XIEFOX/PixDLM](https://github.com/XIEFOX/PixDLM)
**Area**: Semantic Segmentation
**Keywords**: UAV reasoning segmentation, multimodal large language model, dual-path visual encoder, chain-of-thought reasoning, pixel-level prediction

## TL;DR

This paper formally defines the UAV Reasoning Segmentation task, constructs the DRSeg benchmark comprising 10K high-resolution UAV images with chain-of-thought reasoning annotations, and proposes the dual-path pixel-level multimodal large language model PixDLM as a strong baseline.

## Background & Motivation

**Background**: Reasoning Segmentation aims to identify regions in an image satisfying conditions described by free-form textual instructions. Models such as LISA and PixelLM have demonstrated the capacity of multimodal large language models (MLLMs) for implicit reasoning and pixel-level segmentation in ground-view scenarios.

**Limitations of Prior Work**: Existing reasoning segmentation models and datasets are predominantly built upon ground-view or nadir-view imagery, whose visual assumptions—moderate resolution, limited scale variation, stable camera orientation, and relatively large object sizes—are fundamentally inapplicable to UAV imagery. UAV images present three unique challenges: (1) high-altitude oblique perspectives continuously alter projective geometry; (2) extreme scale variation and densely packed small objects, with many critical targets spanning only tens of pixels; and (3) ultra-high-resolution scenes requiring simultaneous reasoning over global semantics and fine-grained high-frequency details.

**Key Challenge**: Existing MLLMs typically employ low-resolution visual tokenization, causing fine-grained UAV details to be lost during compression. Moreover, the absence of a reasoning segmentation benchmark specifically tailored to UAV scenarios impedes systematic research progress.

**Goal**: (1) Formally define the UAV Reasoning Segmentation task and construct a dedicated benchmark dataset; (2) propose a baseline model capable of jointly handling global semantics and local details.

**Key Insight**: The semantic reasoning requirements of UAV imagery are organized along three dimensions—spatial reasoning, attribute reasoning, and scene-level reasoning—corresponding to positional relationships, visual states, and global context, respectively.

**Core Idea**: A dual-path visual encoder (global low-resolution path + high-resolution structural path) is employed to preserve small-object and boundary cues, which are then combined with LLM-driven reasoning for pixel-level segmentation.

## Method

### Overall Architecture

PixDLM consists of four core components: (1) a dual-path visual encoder that extracts global semantic and fine-grained structural features; (2) a MultiPath Alignment module that fuses the dual-path features; (3) an LLM that performs instruction-conditioned reasoning; and (4) a multi-scale decoder that reconstructs the final segmentation mask. Given a UAV image and a natural language instruction, the model outputs a pixel-level mask satisfying the instruction.

### Key Designs

1. **Dual-Path Vision Encoder**:

    - **Function**: Simultaneously captures global semantic context and high-resolution structural details.
    - **Mechanism**: The global path employs a CLIP visual encoder to process low-resolution inputs for semantic features; the structural path employs a SAM encoder to process high-resolution inputs, preserving small-object and boundary cues. The two paths are complementary—CLIP excels at semantic understanding while SAM excels at fine-grained structural perception.
    - **Design Motivation**: A single low-resolution encoder loses densely distributed small-object information in UAV imagery, whereas a single high-resolution encoder incurs prohibitive computational costs. The dual-path design balances semantic understanding with detail preservation.

2. **MultiPath Alignment Module**:

    - **Function**: Lightweight fusion of global semantic and local structural features.
    - **Mechanism**: The semantic features from CLIP and the structural features from SAM are aligned into a unified representation space via a controlled integration scheme, enabling subsequent LLM reasoning to leverage both.
    - **Design Motivation**: The two paths produce features at different scales and semantic levels; an effective alignment mechanism is required for the LLM to simultaneously exploit the advantages of both paths.

3. **DRSeg Dataset Construction Pipeline**:

    - **Function**: Provides 10K high-resolution UAV images with corresponding reasoning annotations.
    - **Mechanism**: The construction follows four stages—manual selection of complex scene images → coarse mask generation via SAM2 followed by human refinement → GPT-5 generation of three-dimensional reasoning QA pairs (with CoT reasoning chains) conditioned on image, mask, and category → human review. The data are uniformly distributed across three reasoning dimensions: spatial, attribute, and scene (each approximately 33.3%).
    - **Design Motivation**: Existing UAV datasets lack fine-grained annotations and reasoning-oriented textual supervision, making them insufficient to support systematic reasoning segmentation research.

### Loss & Training

The model follows the standard LISA training paradigm, employing mask tokens and an embedding-as-mask decoder. Supervised fine-tuning (SFT) mode is supported.

## Key Experimental Results

### Main Results

| Model | Attribute gIoU | Scene gIoU | Spatial gIoU |
|-------|---------------|------------|-------------|
| LISA-13B (zero-shot) | 52.65 | 47.08 | 42.85 |
| PixelLM-7B (zero-shot) | 46.87 | 43.07 | 41.28 |
| LISA-7B (SFT) | 59.22 | 54.45 | 57.33 |
| **PixDLM (Ours)** | **62.80** | **61.75** | **62.51** |

### Ablation Study

| Configuration | Attr gIoU | Scene gIoU | Spatial gIoU |
|---------------|-----------|------------|-------------|
| DRSeg + RRSIS-D + CoT | 61.13 | 55.60 | 60.55 |
| DRSeg + CoT (w/o RRSIS-D) | **62.80** | **61.75** | **62.51** |
| DRSeg (w/o CoT) | 62.51 | 61.67 | 61.98 |

### Key Findings

- PixDLM consistently outperforms both zero-shot and SFT baselines across all three reasoning dimensions, with particularly notable gains in scene-level reasoning (+7.3 vs. SFT LISA).
- Incorporating RRSIS-D data degrades performance, indicating that domain-specific UAV data is more critical than data volume.
- CoT reasoning supervision yields relatively modest gains, suggesting that the model is robust to noisy reasoning chains.

## Highlights & Insights

- **Clear Task Definition**: The semantic requirements of UAV reasoning segmentation are systematically organized into spatial, attribute, and scene dimensions, providing a clear framework for future research.
- **Mature Data Construction Pipeline**: The semi-automatic annotation pipeline combining GPT-5 generation with human review achieves a favorable balance between quality and scalability.
- **Concise and Effective Dual-Path Design**: Leveraging off-the-shelf CLIP and SAM encoders avoids the need to train a high-resolution encoder from scratch.

## Limitations & Future Work

- Approximately 58% of instances are small objects (area < 2%), leaving substantial room for improvement on extremely small targets.
- Only a single target instance is annotated per image, precluding evaluation of multi-object reasoning scenarios.
- The computational overhead of the dual-path encoder is non-trivial and may be insufficient for real-time UAV applications.
- The dataset scale (10K images) is relatively limited; larger-scale data may yield further performance improvements.

## Related Work & Insights

- **vs. LISA**: LISA relies on a single CLIP encoder, whereas PixDLM adds a SAM high-resolution path, yielding clear advantages in UAV small-object scenarios.
- **vs. GeoPix/GeoPixel**: These remote sensing models exploit geographic priors but lack open-vocabulary reasoning capability and perform poorly on densely packed small objects.
- **vs. LLaVA-HR**: LLaVA-HR similarly adopts a dual-path strategy for high-resolution processing, but PixDLM is specifically designed for pixel-level output.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First formal definition of the UAV reasoning segmentation task; the dataset and task formulation are pioneering contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive multi-baseline comparisons with well-designed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Task definition and dataset construction are described in thorough and clear detail.
- **Value**: ⭐⭐⭐⭐ Provides an important benchmark and baseline for UAV visual understanding research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] AnchorSeg: Language Grounded Query Banks for Reasoning Segmentation](../../ACL2026/segmentation/anchorseg_language_grounded_query_banks_for_reasoning_segmentation.md)
- [\[CVPR 2026\] SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](sgma_semantic-guided_modality-aware_segmentation_for_remote_sensing_with_incompl.md)
- [\[CVPR 2026\] RecycleLoRA: Rank-Revealing QR-Based Dual-LoRA Subspace Adaptation for Domain Generalized Semantic Segmentation](recyclelora_rank-revealing_qr-based_dual-lora_subspace_adaptation_for_domain_gen.md)
- [\[CVPR 2026\] Brewing Stronger Features: Dual-Teacher Distillation for Multispectral Earth Observation](brewing_stronger_features_dual-teacher_distillation_for_multispectral_earth_obse.md)
- [\[CVPR 2026\] VIRST: Video-Instructed Reasoning Assistant for SpatioTemporal Segmentation](virst_video-instructed_reasoning_assistant_for_spatiotemporal_segmentation.md)

</div>

<!-- RELATED:END -->
