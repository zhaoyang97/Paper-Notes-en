---
title: >-
  [Paper Note] A Paradigm Shift: Fully End-to-End Training for Temporal Sentence Grounding in Videos
description: >-
  [CVPR 2026][Model Compression][Temporal sentence grounding] This paper proposes the first fully end-to-end framework for Temporal Sentence Grounding in Videos (TSGV). A Sentence-Conditioned Adapter (SCADA) is introduced…
tags:
  - "CVPR 2026"
  - "Model Compression"
  - "Temporal sentence grounding"
  - "end-to-end training"
  - "sentence-conditioned adapter"
  - "vision-language alignment"
  - "TSGV"
date: 2026-05-08
content_hash: bfba2e20fef445ee
---

# A Paradigm Shift: Fully End-to-End Training for Temporal Sentence Grounding in Videos

**Conference**: CVPR 2026
**arXiv**: [2604.02860](https://arxiv.org/abs/2604.02860)  
**Code**: Coming soon  
**Area**: Model Compression
**Keywords**: Temporal sentence grounding, end-to-end training, sentence-conditioned adapter, vision-language alignment, TSGV

## TL;DR
This paper proposes the first fully end-to-end framework for Temporal Sentence Grounding in Videos (TSGV). A Sentence-Conditioned Adapter (SCADA) is introduced to inject sentence embeddings into intermediate layers of the video backbone, dynamically modulating visual features. Combined with a video-centric learning strategy to accelerate training, the method surpasses state-of-the-art performance on Charades-STA and ActivityNet.

## Background & Motivation

**Background**: TSGV aims to localize the temporal segment in an untrimmed video corresponding to a natural language query. Most existing methods employ frozen pre-trained video encoders (e.g., C3D/I3D) for feature extraction and train only the grounding module.

**Limitations of Prior Work**: (1) Video backbones are trained for visual classification but applied to TSGV—a task mismatch exists; (2) Pre-trained models learn only phrase-level object/action concepts and struggle to understand complex natural language semantics; (3) Some methods do not leverage sentence features during the grounding stage, resulting in insufficient cross-modal alignment.

**Key Challenge**: Frozen backbones → features cannot adapt to the TSGV task → limited grounding accuracy. However, directly fine-tuning large backbones incurs substantial memory overhead and risks catastrophic forgetting.

**Key Insight**: Design a lightweight adapter that achieves sentence-conditioned backbone adaptation while fine-tuning only a minimal number of parameters.

**Core Idea**: SCADA injects sentence embeddings into each backbone layer via inner and outer branches, enabling sentence-guided visual feature extraction. A video-centric learning strategy allows multiple queries for the same video to share the feature extraction pass.

## Method

### Overall Architecture
Sentence encoding (DistilBERT) → Video encoding (ViT/I3D/C3D) + **SCADA adapter** → Cross-modal fusion → Detection head (BiLSTM + sentence modulation) → Temporal boundary prediction.

### Key Designs

1. **Sentence-Conditioned Adapter (SCADA)**:

    - Function: Inserted between backbone layers to dynamically modulate visual features using sentence embeddings.
    - **Inner Branch**: Dimensionality reduction → multiplicative sentence modulation → depthwise separable 1D convolution (capturing temporal context) → dimensionality expansion + residual. Output is passed to the next backbone layer.
    - **Outer Branch**: 3D convolution for dimensionality reduction + spatial compression → sentence modulation → 3D convolution for dimensionality expansion → spatial pooling. Output bypasses subsequent backbone layers and is directly aggregated.
    - Final output: $F = \text{Normalize}(x_b + \sum_{i=1}^{n} x_{outer}^i)$
    - Design Motivation: The inner branch enables each backbone layer to perceive sentence semantics; the outer branch extracts query-guided multi-scale features. Only a small number of adapter parameters are trained while the backbone is frozen, resolving memory and forgetting issues.

2. **Video-Centric Learning Strategy**:

    - Function: Accelerates end-to-end training.
    - Mechanism: All queries associated with the same video are grouped into the same mini-batch, so the backbone extracts video features only once and shares them across multiple queries.
    - Design Motivation: Standard sampling causes the same video to repeatedly pass through the backbone, leading to severe computational redundancy. Video-centric sampling also enables the network to align a single video with diverse linguistic contexts within one iteration.

3. **Sentence Fusion in the Detection Head**:

    - BiLSTM architecture with residual connections to capture temporal dependencies.
    - Sentence embeddings are repeatedly fused via element-wise multiplication during detection, achieving deep integration of both modalities.

### Loss & Training
$\mathcal{L} = \mathcal{L}_b + \mathcal{L}_{iou} + \mathcal{L}_{offset}$: boundary probability loss (balanced BCE for positive/negative samples) + IoU loss (classification + L2 regression) + offset loss (Smooth L1).

## Key Experimental Results

### Main Results

| Backbone | Method | Charades R1@0.5 | Charades R1@0.7 | ActivityNet R1@0.5 | ActivityNet mIoU |
|----------|--------|-----------------|-----------------|--------------------|----|
| C3D | MS-2D-TAN | 41.10 | 23.25 | 46.16 | - |
| C3D | APGN | 48.20 | 29.37 | - | - |
| C3D | **Ours** | **50.44** | - | - | - |
| I3D | PGSR et al. | ~53 | ~30 | ~48 | ~48 |
| I3D | **Ours** | **Rank 1** | **Rank 1** | **Leading** | **Leading** |

Charades-STA: R1@0.5 = **48.1%** (ViT); ActivityNet: R1@0.5 = **30.5%**.

### Ablation Study

| Configuration | Charades R1@0.5 | Note |
|---------------|-----------------|------|
| Frozen backbone (baseline) | Baseline | Standard frozen paradigm |
| Full end-to-end fine-tuning | +Large gain | Validates E2E effectiveness |
| + SCADA | +Further gain | Value of sentence conditioning |
| + Video-centric learning | Training speedup | Reduces redundant computation |
| w/o outer branch | Degraded | Multi-scale features are important |
| w/o inner branch | Degraded | Layer-wise modulation is important |

### Key Findings
- End-to-end training yields an **average gain of 16%** over frozen baselines, consistently across different backbones and datasets.
- SCADA improves Charades R1@0.5 from ~38 to ~53 on the I3D backbone, a substantial improvement.
- The potential of ViT as a video encoder for TSGV is fully explored for the first time.
- Video-centric learning accelerates training by several times (depending on the number of queries per video).

## Highlights & Insights
- **Systematic validation of the end-to-end paradigm**: The first work to systematically validate the substantial benefit of end-to-end training for TSGV across multiple backbones (C3D/I3D/ViT-S/B/g), challenging the default assumption that frozen backbones are sufficient.
- **Elegant design of SCADA**: The dual-branch structure enables sentence information to influence both the internal backbone layers (inner branch) and produce skip connections (outer branch), achieving deep cross-modal fusion with minimal additional parameters.
- **Practical value of video-centric learning**: This strategy exploits the natural characteristic of TSGV datasets—multiple queries per video—offering a high-return engineering optimization at low overhead.

## Limitations & Future Work
- Evaluation is limited to Charades-STA and ActivityNet; additional datasets (e.g., TACoS, DiDeMo) remain unexplored.
- The insertion positions and number of SCADA modules are manually configured; automatic search for optimal configurations warrants investigation.
- Comparison with Video LLM-based methods (e.g., D2VLM R1@0.5 = 50.30) is not yet comprehensive.
- Training time and GPU memory consumption for the ViT backbone are not reported in detail.

## Related Work & Insights
- **vs. 2D-TAN/APGN et al.**: These methods freeze the backbone and train only the grounding module; this paper demonstrates that end-to-end training is a superior paradigm.
- **vs. Video LLMs**: LLM-based methods predict timestamps via temporally sensitive instruction tuning; this paper achieves comparable performance on certain metrics without relying on LLMs.
- **vs. Other adapter methods**: General-purpose adapters such as LoRA do not consider sentence conditioning; SCADA is specifically designed for cross-modal tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ SCADA is a creative design; while the E2E paradigm is not entirely novel, its systematic validation is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-backbone evaluation and thorough ablations; dataset coverage could be broader.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear and method description is complete.
- Value: ⭐⭐⭐⭐ Provides a new training paradigm reference for the TSGV community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Partial Forward Blocking: A Novel Data Pruning Paradigm for Lossless Training Acceleration](../../ICCV2025/model_compression/partial_forward_blocking_a_novel_data_pruning_paradigm_for_lossless_training_acc.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[ICLR 2026\] Grounding and Enhancing Informativeness and Utility in Dataset Distillation](../../ICLR2026/model_compression/grounding_and_enhancing_informativeness_and_utility_in_dataset_distillation.md)
- [\[CVPR 2026\] Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge](critical_patch-aware_sparse_prompting_with_decoupled_training_for_continual_lear.md)
- [\[ICLR 2026\] Why Attention Patterns Exist: A Unifying Temporal Perspective Analysis](../../ICLR2026/model_compression/why_attention_patterns_exist_a_unifying_temporal_perspective_analysis.md)

</div>

<!-- RELATED:END -->
