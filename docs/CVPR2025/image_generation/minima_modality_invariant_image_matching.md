---
title: >-
  [Paper Note] MINIMA: Modality Invariant Image Matching
description: >-
  [CVPR 2025][Image Generation][Cross-modal Image Matching] MINIMA proposes a unified cross-modal image matching framework. By designing a data engine to generate a multimodal synthetic dataset MD-syn (480M pairs) from cheap RGB image pairs, any existing matching pipeline can obtain cross-modal matching capability through simple fine-tuning, significantly outperforming modality-specific methods across 19 cross-modal scenarios.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Cross-modal Image Matching"
  - "Data Engine"
  - "Synthetic Data"
  - "Modality Invariant"
  - "Feature Matching"
date: 2026-05-08
content_hash: ecda9930c05531e4
---

# MINIMA: Modality Invariant Image Matching

**Conference**: CVPR 2025  
**arXiv**: [2412.19412](https://arxiv.org/abs/2412.19412)  
**Code**: [https://github.com/LSXI7/MINIMA](https://github.com/LSXI7/MINIMA)  
**Area**: Image Matching / Multimodal Perception  
**Keywords**: Cross-modal Image Matching, Data Engine, Synthetic Data, Modality Invariant, Feature Matching

## TL;DR
MINIMA proposes a unified cross-modal image matching framework. By designing a data engine to generate a multimodal synthetic dataset MD-syn (480M pairs) from cheap RGB image pairs, any existing matching pipeline can obtain cross-modal matching capability through simple fine-tuning, significantly outperforming modality-specific methods across 19 cross-modal scenarios.

## Background & Motivation

1. **Background**: Image matching is fundamental to various applications such as visual localization and object detection. There are existing mature sparse/semi-dense/dense matching methods (LightGlue, LoFTR, RoMa, etc.) for RGB image matching, benefiting from large-scale datasets such as MegaDepth and ScanNet.
2. **Limitations of Prior Work**: Cross-modal matching (RGB-IR, RGB-depth, RGB-event, etc.) faces severe domain discrepancy challenges. Existing cross-modal datasets are extremely small in scale (thousands of pairs vs. tens of millions of pairs in RGB datasets), cover limited scenarios, and have time-consuming and expensive annotations. Consequently, modality-specific local feature methods possess poor generalization capabilities outside of the training set scenarios.
3. **Key Challenge**: The essence of the problem lies in the **data gap**—multimodal image acquisition requires multiple different imaging devices to capture the same scene simultaneously, which is costly and makes scene diversity difficult to guarantee; annotations also cannot be directly obtained using tools like COLMAP.
4. **Goal**: How can a unified model handle all cross-modal matching? How can large-scale, high-quality multimodal matching training data be obtained at low cost?
5. **Key Insight**: Starting from cheap and abundant RGB image pairs, generative models are leveraged to convert RGB images into pseudo-images of other modalities. Since the conversion is pixel-to-pixel, the original matching annotations (such as depth and pose) from the RGB pairs can be directly inherited by the generated multimodal data.
6. **Core Idea**: To expand RGB image pairs from MegaDepth into 6 modalities (infrared, depth, event, normal, oil painting, sketch) via a data engine, generating 480M cross-modal pairs to train any matching pipeline.

## Method

### Overall Architecture
MINIMA consists of two parts: (1) **Data Engine**: generates synthetic images of 6 modalities from MegaDepth's RGB image pairs to construct the MD-syn dataset; (2) **Model Training**: pre-trains the matching model on RGB, and then fine-tunes it on MD-syn using randomly selected cross-modal pairs. The output is a unified matching model capable of handling any modality combination.

### Key Designs

1. **Cross-Modal Data Engine**:

    - **Function**: Translates RGB images into synthetic images of multiple modalities automatically while inheriting original matching annotations.
    - **Mechanism**: Utilizes 6 generative models to convert RGB to: infrared (LoRA fine-tuned on StyleBooth using LLVIP+M3FD), depth (Depth Anything V2), event (physics-based simulator modeling brightness changes), normal (DSINE), oil painting (Paint Transformer), and sketch (Anime2Sketch). For a pair of RGB images $(A_0, B_0)$, two sets of K=6 modality image series are generated, which theoretically construct 480M cross-modal image pairs.
    - **Design Motivation**: Three major advantages—Cheap (easy access to RGB images), Flexible (freely controllable generation scale and modality balance), and High-quality (generated image resolution is consistent with original RGB, and annotations are directly inherited).

2. **Pre-train + Fine-tune**:

    - **Function**: Enables existing matching pipelines to efficiently acquire cross-modal capabilities.
    - **Mechanism**: Stage 1 pre-trains the matching model on RGB data until convergence (or directly uses official pre-trained weights); Stage 2 fine-tunes on MD-syn with randomly selected cross-modal pairs using a small learning rate. Randomly choosing the modality pair prevents the model from biasing toward a specific modality combination.
    - **Design Motivation**: Training multimodal data from scratch is hard to converge (due to large inter-modality variance), while RGB pre-training provides solid matching priors. Fine-tuning allows fast convergence and cross-modal generalization.

3. **Multi-Pipeline Adaptation**:

    - **Function**: Demonstrates the universality of the data engine—it does not rely on specific network architectures.
    - **Mechanism**: Selects three representative matching pipelines—the sparse matcher LightGlue, semi-dense matcher LoFTR, and dense matcher RoMa—fine-tunes them individually, and releases MINIMA_LG, MINIMA_LoFTR, and MINIMA_RoMa.
    - **Design Motivation**: Shows that MINIMA is a model-agnostic, data-level solution from which any advanced matching method can benefit.

### Loss & Training
The original loss functions of individual baseline matching pipelines are directly adopted, employing a smaller learning rate during fine-tuning. The infrared modality generator is fine-tuned on LLVIP+M3FD with $lr=1\times10^{-4}$ for 210k steps and LoRA rank 256.

## Key Experimental Results

### Main Results

Overall matching accuracy (AUC@10° or @10px) on 6 real-world cross-modal datasets:

| Method | Type | RGB-IR | RGB-Depth | RGB-Normal | RGB-Event | RGB-Sketch | RGB-Paint |
|------|------|--------|-----------|------------|-----------|------------|-----------|
| LightGlue | Sparse | 17.73 | 2.87 | 24.93 | 22.40 | 44.47 | 27.99 |
| **MINIMA_LG** | Sparse | **30.24** | **32.53** | **37.33** | **36.27** | **45.71** | **32.85** |
| LoFTR | Semi-dense | 12.58 | 0.44 | 12.07 | 12.43 | 54.82 | 12.22 |
| **MINIMA_LoFTR** | Semi-dense | **32.36** | **28.81** | **44.26** | **32.74** | **53.54** | **15.45** |
| RoMa | Dense | 29.46 | 0.38 | 39.28 | 18.14 | 72.25 | 44.73 |
| **MINIMA_RoMa** | Dense | **46.77** | **42.17** | **50.87** | **44.32** | **73.10** | **50.34** |

The improvement on RGB-Depth is especially striking (LightGlue: 2.87→32.53).

### Ablation Study

| Data Strategy | AUC@10° (Average) | Explanation |
|---------|---------------|------|
| RGB pre-training only | ~20 | Baseline, poor cross-modal accuracy |
| + Single-modality fine-tuning | ~30 | Only effective for the trained modalities |
| + Random multi-modality fine-tuning (MD-syn) | ~38 | Comprehensively improves all modalities |

### Key Findings
- **The data engine is the core contribution**: For the exact same matching architecture, using MD-syn data for training leads to a massive boost in cross-modal performance, demonstrating that the bottleneck lies in the data rather than the model.
- **Excellent zero-shot cross-modal generalization**: It significantly outperforms methods tailored specifically for single modalities (such as ReDFeat and XoFTR) even on unseen real-world cross-modal datasets.
- **All types of matching pipelines benefit**: Significant improvements are observed across all three types (sparse, semi-dense, dense), proving the universality of the data engine.
- Data balance among modalities is crucial—randomly selecting modality pairs prevents overfitting to simpler modalities.

## Highlights & Insights
- **The "data as methodology" paradigm** is extremely elegant: instead of designing modality-specific modules, the general matcher is endowed with cross-modal capabilities simply by augmenting training data. This concept is highly transferable to various cross-domain tasks—constructing large-scale pseudo-domain data first, then fine-tuning the strongest existing models.
- **The layout of inheriting annotations through synthetic data** is ingenious: pixel-wise style transfer does not affect geometric relationships, meaning depth and pose annotations can be inherited at zero cost. This solves the fundamental challenge of annotating multimodal data.
- Evaluates 19 cross-modal matching scenarios under a unified framework for the first time, establishing a comprehensive evaluation benchmark.

## Limitations & Future Work
- The synthetic pseudo-modalities still have domain gaps relative to real-world modalities (e.g., the synthesis quality of infrared is limited by the size of the fine-tuning data), which may be inaccurate in certain challenging scenarios.
- Only using MegaDepth (outdoor) as the base dataset may limit the performance of cross-modal matching in indoor scenarios.
- Only 6 modalities are currently supported; more specialized modalities like SAR or medical imaging are not covered.
- The computational overhead of the generative models is quite high (especially the diffusion-based infrared generator), meaning dataset construction costs are non-negligible.

## Related Work & Insights
- **vs. ReDFeat**: Trains detectors + descriptors individually for each modality, and trains/tests separately on each dataset, leading to poor generalization. MINIMA processes all modalities within a unified model with superior performance.
- **vs. XoFTR**: Only designs a two-stage training strategy for RGB-IR, requiring modality-specific matching rules. MINIMA outperforms it on RGB-IR while covering a much broader scope.
- **vs. GIM**: GIM also attempts to expand training data from videos to improve generalization, but yields only minor gains despite using several times more images. MINIMA's approach of expanding along the modality dimension is significantly more cost-effective.

## Rating
- Novelty: ⭐⭐⭐⭐ The data engine concept is simple yet highly effective, demonstrating for the first time that synthetic multimodal data can replace real-world data for training matchers.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 19 cross-modal scenarios, 3 matching paradigms, and includes an extensive suite of ablation and zero-shot experiments.
- Writing Quality: ⭐⭐⭐⭐ Solid clarity in motivation and systematically organized experiments.
- Value: ⭐⭐⭐⭐⭐ High value to the community via both the data engine and the MD-syn dataset.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Not Just Text: Uncovering Vision Modality Typographic Threats in Image Generation Models](not_just_text_uncovering_vision_modality_typographic_threats_in_image_generation.md)
- [\[CVPR 2026\] Semantic Alignment for Pose-Invariant Identity Preserving Diffusion](../../CVPR2026/image_generation/semantic_alignment_for_pose-invariant_identity_preserving_diffusion.md)
- [\[ICCV 2025\] Bridging the Skeleton-Text Modality Gap: Diffusion-Powered Modality Alignment for Zero-shot Skeleton-based Action Recognition](../../ICCV2025/image_generation/bridging_the_skeleton_text_modality_gap_diffusion_powered_modality_alignment_for.md)
- [\[NeurIPS 2025\] Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge](../../NeurIPS2025/image_generation/towards_general_modality_translation_with_contrastive_and_predictive_latent_diff.md)
- [\[CVPR 2025\] Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment](diff2flow_training_flow_matching_models_via_diffusion_model_alignment.md)

</div>

<!-- RELATED:END -->
