---
title: >-
  [Paper Note] Multigranular Evaluation for Brain Visual Decoding
description: >-
  [AAAI 2026][Segmentation][brain decoding] This paper proposes BASIC, a multigranular evaluation framework that unifies the assessment of brain visual decoding quality along two axes — structural (multi-level segmentation mask matching across four granularities) and semantic (precision/recall/F1 over object/attribute/relation graphs extracted by MLLMs) — covering six modality combinations of fMRI/EEG × Image/Video/3D, thereby addressing the limitations of metric saturation, lack of neuroscientific grounding, and insufficient fine-grained diagnostic capacity in existing evaluation protocols.
tags:
  - AAAI 2026
  - Segmentation
  - brain decoding
  - evaluation metric
  - semantic matching
  - MLLM
date: 2026-05-08
content_hash: f1d304d4bd2c0aea
---

# Multigranular Evaluation for Brain Visual Decoding

**Conference**: AAAI 2026
**arXiv**: [2507.07993](https://arxiv.org/abs/2507.07993)
**Code**: [GitHub](https://github.com/weihaox/BASIC)
**Area**: Image Segmentation
**Keywords**: brain decoding, evaluation metric, segmentation, semantic matching, MLLM

## TL;DR

This paper proposes BASIC, a multigranular evaluation framework that unifies the assessment of brain visual decoding quality along two axes — structural (multi-level segmentation mask matching across four granularities) and semantic (precision/recall/F1 over object/attribute/relation graphs extracted by MLLMs) — covering six modality combinations of fMRI/EEG × Image/Video/3D, thereby addressing the limitations of metric saturation, lack of neuroscientific grounding, and insufficient fine-grained diagnostic capacity in existing evaluation protocols.

## Background & Motivation

Brain visual decoding has achieved the reconstruction of images, videos, and even 3D shapes from fMRI/EEG neural signals, yet the evaluation framework has lagged substantially behind methodological advances. Three core limitations exist:

First, **metric saturation** — mainstream metrics such as PixCorr, SSIM, and CLIP yield converging scores across state-of-the-art models, making it impossible to discriminate decoding quality. For instance, multiple methods achieve nearly identical CLIP scores despite notable differences in semantic accuracy.

Second, **lack of neuroscientific grounding** — human visual perception is hierarchical, progressing from attention-driven salient object recognition to attribute perception, spatial relationship understanding, and scene-level semantic coherence. Existing metrics do not reflect this multilevel structure and cannot determine whether decoded details originate from genuine neural signals or are hallucinated by generative models.

Third, **absence of diagnostic capacity** — black-box single-score metrics cannot inform researchers where reconstruction fails: incorrect object categories, wrong attributes, or implausible spatial relations.

The paper's **Starting Point** is to design a unified evaluation framework, BASIC, that simultaneously covers low-level structural and high-level semantic aspects with multigranular diagnostic capability, applicable to all stimulus–neuroimaging modality combinations.

## Method

### Overall Architecture

BASIC (**B**rain-**A**ligned **S**tructural, **I**nferential, and **C**ontextual similarity) comprises two complementary sub-metrics:
- **BASIC-L**: Low-level structural similarity — multigranular structural matching based on four-level segmentation masks.
- **BASIC-H**: High-level semantic similarity — combining inferential (object/attribute/relation matching) and contextual (scene narrative coherence) components.

### Key Designs

1. **Five-Dimensional Evaluation Framework**

    - **Function**: Defines the perceptual dimensions that brain decoding evaluation should cover.
    - **Mechanism**: Scene (layout/geometry/events/style), Object (category/generality/specificity), Attribute (appearance color/texture/position/quantity/text symbols), Relation (spatial/part-whole/interaction/motion), and Camera (illumination/viewpoint/motion).
    - **Design Motivation**: Grounded in visual neuroscience and cognitive psychology research, aligned with the hierarchical structure of human visual perception, and consistent with the scene understanding structure of multimodal large language models.

2. **BASIC-L: Multigranular Segmentation Matching**

    - **Function**: Quantifies spatial structural consistency between reconstructed and reference images.
    - **Mechanism**: Mask correspondence matching is performed at four segmentation granularities: Foreground (salient foreground detection) → Semantic (semantic categories) → Instance (instance-level) → Part (component-level). Both reconstructed and reference images undergo multigranular segmentation, and IoU and AP are computed via granularity-aware mask correspondence.
    - **Design Motivation**: Single-granularity segmentation matching may omit critical information — foreground segmentation only captures object presence, semantic segmentation ignores instance distinction, and instance segmentation ignores part-level structure. The coarse-to-fine hierarchical matching provides comprehensive coverage of spatial structural fidelity.

3. **BASIC-H: Structured Semantic Matching**

    - **Function**: Quantifies the high-level semantic correspondence between reconstructed and reference images.
    - **Mechanism**: A three-step pipeline — (1) an MLLM (e.g., GPT-4V) generates detailed structured descriptions for both reconstructed and reference images; (2) descriptions are parsed into semantic graphs, extracting object sets, attribute sets, and relation triples; (3) Precision/Recall/F1 are computed separately for Object, Attribute, and Relation, and aggregated into the BASIC-H score.
    - **Design Motivation**: Traditional feature similarity (cosine distance of CLIP embeddings) compresses multidimensional semantics into a single score, making it impossible to distinguish cases such as "correct objects but wrong attributes" from "correct object count but confused categories." Structured semantic matching provides interpretable diagnostic information.

### Loss & Training

BASIC is an evaluation metric rather than a training method and does not involve loss function design. The core components of the framework — a pretrained segmentation model (for BASIC-L) and an MLLM (for BASIC-H) — are both used in a frozen manner.

## Key Experimental Results

### Main Results

**BASIC-H scores on the NSD dataset (fMRI→Image)**:

| Method | Object F1 | Attribute F1 | Relation F1 | BASIC-H |
|--------|-----------|-------------|-------------|---------|
| SDRecon | 53.79 | 14.96 | 39.06 | 35.31 |
| BrainDiffuser | 58.09 | 19.43 | 43.50 | 39.71 |
| MindEye | 61.26 | 25.06 | 48.84 | 44.30 |
| DREAM | 63.56 | 25.92 | 52.91 | 46.37 |
| MindEye2 | 61.72 | 24.71 | 49.07 | 44.39 |
| NeuroVLA | **64.57** | **28.65** | **52.95** | **47.88** |
| STTM | 62.88 | 26.64 | 50.36 | 45.88 |
| MindTuner | 61.95 | 24.73 | 49.80 | 44.63 |
| BrainGuard | 62.43 | 25.84 | 50.60 | 45.43 |

**Cross-modal BASIC-H comparison**:

| Dataset (Modality) | Best Method | BASIC-H |
|--------------------|------------|---------|
| NSD (fMRI→Image) | NeuroVLA | 47.88 |
| CC2017 (fMRI→Video) | NeuroClips | 45.12 |
| SEED-DV (EEG→Video) | EEG2Video | 49.54 |
| EEG-Things (EEG→Image) | ATM | 30.55 |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| BASIC-H per dimension | Attribute F1 consistently low (14–28) | Attribute reconstruction is the weakest aspect of brain decoding |
| Object vs. Relation | Relation F1 < Object F1 | Inter-object relations are harder to reconstruct than objects themselves |
| BASIC-L NeuroPictor | 25.88 (highest) | Structural ranking differs from BASIC-H ranking |

### Key Findings

- BASIC-H maintains good discriminability across state-of-the-art methods (35.31 to 47.88), whereas conventional CLIP scores have saturated.
- **Attribute is the most significant weakness in brain decoding**: Attribute F1 does not exceed 28.65 across all methods, far below Object and Relation scores.
- **Structural ranking ≠ semantic ranking**: NeuroPictor ranks highest on BASIC-L, while NeuroVLA leads on BASIC-H, demonstrating that the two dimensions capture distinct aspects.
- EEG-Image decoding scores are substantially lower than fMRI-Image scores (30.55 vs. 47.88), quantifying the information gap between the two neuroimaging modalities.
- BASIC uniformly covers all six stimulus–neuroimaging modality combinations (fMRI/EEG × Image/Video/3D), making it the first framework of such breadth.

## Highlights & Insights

- **First unified cross-modal evaluation framework**: The same metric applies to all combinations of fMRI/EEG × Image/Video/3D, enabling cross-modal comparison for the first time.
- **The finding that "Attribute is a blind spot in brain decoding"** carries significant practical guidance: future methods should focus on improving the reconstruction of color, texture, and material attributes.
- **Using MLLMs for automated semantic evaluation** is an elegant design choice: it avoids the bottleneck of manual annotation required by traditional methods and can improve automatically as MLLM capabilities advance.
- **The finding that structural ranking ≠ semantic ranking** demonstrates the insufficiency of single-dimensional evaluation — a method may achieve high spatial structural fidelity while exhibiting semantic confusion.
- The evaluation dimension taxonomy is theoretically grounded in cognitive neuroscience rather than being an ad hoc combination.

## Limitations & Future Work

- **MLLM hallucination risk**: MLLM-generated descriptions may themselves contain hallucinations, introducing evaluation noise; this is particularly problematic for ambiguous or low-quality reconstructed images.
- **Dependence on segmentation models**: The reliability of BASIC-L is bounded by the accuracy of the underlying segmentation model, especially for non-natural images (e.g., 3D renders, video frames).
- **Lack of human perception correlation validation**: No human correlation study has been conducted to verify whether BASIC scores align with human subjective perceptual judgments.
- **High computational cost**: Running both an MLLM and multi-level segmentation for each image pair incurs non-trivial computational overhead at scale.
- **Semantic graph construction may be incomplete for complex scenes**: Relation triple extraction relies on text parsing and may miss interactions in multi-object scenes.
- **Contextual similarity definition is somewhat vague**: The paper primarily presents Object/Attribute/Relation results for BASIC-H; the quantification of global scene coherence lacks sufficient clarity.

## Related Work & Insights

Compared with the conventional 8-metric protocol (PixCorr/SSIM/AlexNet-2/5/Inception/CLIP/EffNet/SwAV), BASIC provides interpretable multigranular evaluation. Compared with task-specific metrics such as $n$-way classification accuracy, BASIC offers unified applicability across datasets and modalities.

The idea of using MLLMs for automated evaluation is generalizable to image generation/editing quality assessment and semantic consistency evaluation in text-to-image generation. The semantic graph matching approach also provides a valuable reference for evaluating scene graph generation and visual question answering. The multigranular segmentation matching design is likewise worth borrowing for the evaluation of image segmentation quality itself.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First multigranular unified evaluation framework targeting brain decoding; the combination of MLLM and segmentation for evaluation is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 14+ methods and 6 modality combinations, reasonably comprehensive; lacks human correlation study.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with well-reasoned argumentation for the dimension taxonomy.
- **Value**: ⭐⭐⭐⭐ Substantially advances evaluation standardization in the brain decoding field; the discovery of the attribute bottleneck provides actionable guidance.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Comparative Evaluation of Traditional Methods and Deep Learning for Brain Glioma Imaging](../../CVPR2026/segmentation/comparative_evaluation_of_traditional_methods_and_deep_learning_for_brain_glioma.md)
- [\[CVPR 2026\] Comparative Evaluation of Traditional Methods and Deep Learning for Brain Glioma Imaging. Review Paper](../../CVPR2026/segmentation/comparative_evaluation_of_traditional_methods_and.md)
- [\[AAAI 2026\] EAGLE: Episodic Appearance- and Geometry-Aware Memory for Unified 2D-3D Visual Query Localization](eagle_episodic_appearance-_and_geometry-aware_memory_for_unified_2d-3d_visual_qu.md)
- [\[AAAI 2026\] LWGANet: Addressing Spatial and Channel Redundancy in Remote Sensing Visual Tasks with Light-Weight Grouped Attention](lwganet_addressing_spatial_and_channel_redundancy_in_remote_sensing_visual_tasks.md)
- [\[AAAI 2026\] RSVG-ZeroOV: Exploring a Training-Free Framework for Zero-Shot Open-Vocabulary Visual Grounding in Remote Sensing Images](rsvg-zeroov_exploring_a_training-free_framework_for_zero-shot_open-vocabulary_vi.md)

<!-- RELATED:END -->
