---
title: >-
  [Paper Note] Defect Spectrum: A Granular Look of Large-Scale Defect Datasets with Rich Semantics
description: >-
  [ECCV 2024][Signal & Communication][Defect detection] This paper constructs the Defect Spectrum dataset, providing fine-grained, semantic-rich, and large-scale multi-class defect annotations (125 defect classes, 3,518 + 1,920 images) across four industrial benchmarks. It also proposes Defect-Gen, a two-stage diffusion generator, to synthesize high-quality, diverse defect images under few-shot conditions, improving defect segmentation mIoU by up to 9.85.
tags:
  - "ECCV 2024"
  - "Signal & Communication"
  - "Defect detection"
  - "semantic-rich annotation"
  - "dataset benchmark"
  - "defect image generation"
  - "diffusion models"
date: 2026-05-08
content_hash: 80ddac03a6235fab
---

# Defect Spectrum: A Granular Look of Large-Scale Defect Datasets with Rich Semantics

**Conference**: ECCV 2024  
**arXiv**: [2310.17316](https://arxiv.org/abs/2310.17316)  
**Code**: [https://envision-research.github.io/Defect_Spectrum/](https://envision-research.github.io/Defect_Spectrum/)  
**Area**: Signal & Communication  
**Keywords**: Defect detection, semantic-rich annotation, dataset benchmark, defect image generation, diffusion models

## TL;DR

This paper constructs the Defect Spectrum dataset, providing fine-grained, semantic-rich, and large-scale multi-class defect annotations (125 defect classes, 3,518 + 1,920 images) across four industrial benchmarks. It also proposes Defect-Gen, a two-stage diffusion generator, to synthesize high-quality, diverse defect images under few-shot conditions, improving defect segmentation mIoU by up to 9.85.

## Background & Motivation

**Background**: Industrial defect detection is a critical pipeline in closed-loop manufacturing systems. Currently, several industrial defect datasets exist, such as MVTec AD, VISION, and DAGM2007. Mainstream approaches comprise three main tasks: anomaly detection (e.g., PatchCore, PADIM), defect classification, and defect segmentation.

**Limitations of Prior Work**: Existing datasets suffer from three major issues: (1) Insufficient annotation precision—many datasets only provide binary masks; for example, MVTec offers pixel-level annotations but does not distinguish defect types; (2) Inadequate semantic granularity—multiple defects may coexist in one image, but existing annotations categorize them under a single class; (3) Scarcity of defect samples—for instance, DAGM has only 900 defect images and MVTec has only 1,258, which is far smaller than natural image datasets.

**Key Challenge**: Practical industrial applications require precise determination of defect types, locations, and sizes to decide how to handle products (e.g., misaligned zipper teeth require re-work, whereas minor fabric defects can be sold at a discount). However, the annotation granularity of existing datasets cannot support such fine-grained decisions. Meanwhile, defect data is naturally scarce, which constrains the training performance of deep learning models.

**Goal**: (1) Establish a precise, semantic-rich, and large-scale industrial defect annotation benchmark; (2) Design a generative model capable of synthesizing high-quality, diverse defect images under extremely few-shot conditions.

**Key Insight**: The authors start from the annotation quality of existing datasets and re-annotate as well as refine four mainstream benchmarks (MVTec, VISION, DAGM2007, Cotton-Fabric). Concurrently, to address the overfitting issue of diffusion models, they propose restricting the receptive field and adopting a two-stage inference process to achieve diverse generation in few-shot scenarios.

**Core Idea**: Construct a high-quality, fine-grained defect dataset through re-annotation, and solve the few-shot defect generation challenge using a two-stage diffusion model (learning global structure with a large receptive field while generating local diversity with a small receptive field).

## Method

### Overall Architecture

The proposed work is divided into two major components: dataset construction and defect generation. For dataset construction, comprehensive re-annotation is performed on four benchmarks (MVTec, VISION, DAGM2007, and Cotton-Fabric) to provide precise boundaries, multi-class labels, and detailed descriptions. For defect generation, a two-stage diffusion generator named Defect-Gen is introduced, which takes a few real defect image-mask pairs as input and outputs newly synthesized defect images along with their corresponding multi-class masks.

### Key Designs

1. **Annotation Improvements**:

    - **Function**: Elevate the annotation quality of existing datasets to meet the standard required for industrial applications.
    - **Mechanism**: The improvements span three dimensions: precision enhancement (correcting contours, labeling missing defects), semantic enrichment (identifying multiple defect types within a single image, expanding from binary masks to 125 categories), and detailed description (adding text descriptions for each sample to support VLM research). Additionally, an auxiliary labeling tool, Defect-Click (fine-tuned from Focal-Click for the industrial defect domain), is developed to increase labeling efficiency by approximately 60%.
    - **Design Motivation**: Real-world industrial scenarios require distinguishing different defect types and determining treatment protocols based on severity, which binary masks fail to support. It was found that 552 images contain multiple defect types but were overlooked by prior annotations.

2. **Patch-level Distribution Modeling**:

    - **Function**: Resolve the overfitting problem of diffusion models under extremely few-shot conditions.
    - **Mechanism**: Traditional diffusion models suffer from severe overfitting and lack of diversity when sample sizes are small (e.g., 25 images), merely copying training samples. According to Vapnik-Chervonenkis (VC) theory, overfitting is correlated with the ratio of data dimensionality to sample size. By restricting the number of downsampling layers in the U-Net to narrow the receptive field, the model effectively learns distributions on smaller image patches. Under this setting, the data dimension ($h_{patch} \times w_{patch} \times n_{total}$) is substantially reduced while the effective sample size is increased, alleviating overfitting.
    - **Design Motivation**: Directly training image-level diffusion models on few-shot data leads to overfitting because the dimensionality is much larger than the sample size, whereas naive patch cropping fails to retain global image structure. Restricting the receptive field via architectural design implicitly enables patch-level modeling without requiring explicit patch reconstruction steps.

3. **Two-Stage Diffusion Inference**:

    - **Function**: Introduce local diversity while preserving global structure.
    - **Mechanism**: Two diffusion models are trained: one with a large receptive field (standard U-Net) and another with a small receptive field (U-Net with reduced downsampling). During inference, the large model starts from pure noise and runs for a specified number of steps to establish the global geometric structure, then hands over the intermediate result at the switching timestep $u$ to the small model to continue denoising and generate local details. The large model ensures correct global structure (e.g., product shape), while the small model generates diverse local defect details. The hyperparameter $u$ and the receptive field size of the small model control the trade-off between fidelity and diversity.
    - **Design Motivation**: Models with only small receptive fields yield diversity but suffer from global shape distortion (Figure 4b), while models with only large receptive fields generate high-fidelity but lack diversity (Figure 4a). The two-stage strategy combines the strengths of both. This design is inspired by the observation that different timesteps in diffusion models correspond to different levels of representation—early steps generate coarse-grained geometry, whereas later steps generate fine details.

### Loss & Training

The diffusion models are trained using the standard DDPM loss function. A key engineering design is concatenating the defect mask with the image channels ($x = I \oplus M$) as joint inputs, allowing the model to simultaneously generate the image and its corresponding multi-class mask with negligible extra computational overhead. Training is performed on 4 RTX 3090 GPUs with a batch size of 2, a learning rate of $1e{-4}$, and 150,000 iterations. Experiments show that a switching timestep of $u=50$ and a small model with a moderate receptive field yield the best balance between fidelity and diversity.

## Key Experimental Results

### Main Results

| Dataset | Metric | Trained on Ours | Trained on Original | Gain |
|---|---|---|---|---|
| Industrial Simulation | Recall (%) | 96.07 | 85.33 | +10.74% |
| Industrial Simulation | FPR (%) | 16.50 | 49.60 | -33.10% |
| DS-MVTec (DeepLabV3+) | mIoU | 55.55 | 51.58 | +3.97 |
| DS-MVTec (MiT-B0) | mIoU | 56.21 | 46.45 | +9.76 |
| DS-Cotton (DeepLabV3+) | mIoU | 58.58 | 48.73 | +9.85 |
| Original MVTec (Ours vs DDPM) | Mean mIoU | 67.76 | 65.07 | +2.69 |

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| Large receptive field (Standard U-Net) | Low LPIPS, Low FID | High fidelity but low diversity, memorizing the training set |
| Small receptive field (Reduced downsampling) | High LPIPS, High FID | High diversity but distorted global structure |
| Two-stage ($u=50$, Moderate RF) | Moderate LPIPS and FID | Optimal fidelity-diversity trade-off |
| Synthetic data 20%/100%/200%/300% | mIoU change | 100% synthetic data is a reasonable choice, performance degrades beyond 300% |
| Transformer vs CNN model | mIoU gain range | Transformer models (MiT-B0) benefit more from synthetic data |

### Key Findings

- Defect Spectrum's fine-grained annotations improve recall by 10.74% and reduce the false positive rate (FPR) by 33.10% in industrial simulation experiments.
- Transformer models benefit more from synthetic data than CNN models: MiT-B0 improves from 46.45 to 56.21 on DS-MVTec.
- The volume of synthetic data is not "the more the better"; a ratio of approximately 100% is optimal, and performance begins to decline when exceeding 300%.
- DeepLabV3+ demonstrates the most stable performance across multiple datasets, rendering it a robust baseline choice for defect segmentation.
- SAM (Segment Anything Model) underperforms compared to the specifically fine-tuned Defect-Click for industrial defect regions, due to the large domain gap between industrial defects and natural images.

## Highlights & Insights

- Solid methodology for dataset construction: 580 hours of professional annotation, development of auxiliary labeling tools, and comprehensive documentation of annotation improvements.
- The proposed two-stage diffusion generation is simple yet effective; implicit patch-level modeling via architecture-level receptive field limits bypasses explicit patch manipulation.
- The joint image-mask generation approach is highly referenceable, providing paired annotations with virtually zero extra overhead.
- The industrial simulation design successfully demonstrates the practical value of fine-grained annotations.
- It paves the path for VLM applications in industrial inspection (by providing text descriptions).

## Limitations & Future Work

- Although the dataset scale is large, it remains built upon existing benchmarks, meaning product categories are still limited.
- Defect-Gen requires training two models (large and small) separately for each product category, which limits its scalability.
- The selection of switching timestep $u$ and the receptive field size relies on manual hyperparameter tuning.
- Pre-trained large-scale generative models (like Stable Diffusion) are not leveraged as the foundation of Defect-Gen, potentially missing opportunities for transfer learning.
- The quality of text descriptions has not been systematically evaluated, which may affect the validity of subsequent VLM research.
- Refining the defect injection process (e.g., via inpainting) using pre-trained diffusion models could be explored for better diversity.

## Related Work & Insights

- **MVTec AD**: The most widely adopted anomaly detection benchmark, but limited to binary masks. This work expands it to 125 defect classes.
- **DefectGAN**: A GAN-based defect generation method, which still requires a considerable amount of real defect data.
- **SinDiffusion**: Learns a diffusion model from a single image, which can generate diverse samples but suffers from unrealistic structures.
- **Perception Prioritized Training**: Reveals that different timesteps of diffusion models correspond to different levels of representations, inspiring the two-stage strategy in this paper.
- Insight: The concept of combining receptive field limits with multi-stage inference can be applied to other few-shot generation tasks (such as data augmentation in medical imaging or remote sensing).

## Rating

- Novelty: ⭐⭐⭐ Solid dataset construction and annotation pipeline; the two-stage strategy of Defect-Gen is ingenious but not highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluations covering multiple segmentation models, synthetic data ratio analysis, industrial simulation experiments, and comparisons against other generative approaches.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, detailed dataset analysis, and rich illustrative figures/tables.
- Value: ⭐⭐⭐⭐ The dataset holds substantial value for the industrial defect detection community, and Defect-Gen provides a practical solution for few-shot industrial data augmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Large Language Model (LLM)-enabled In-context Learning for Wireless Network Optimization](../../ICML2025/signal_comm/large_language_model_llm-enabled_in-context_learning_for_wireless_network_optimi.md)
- [\[ECCV 2024\] PYRA: Parallel Yielding Re-Activation for Training-Inference Efficient Task Adaptation](pyra_parallel_yielding_re-activation_for_training-inference_efficient_task_adapt.md)
- [\[ECCV 2024\] QueryCDR: Query-based Controllable Distortion Rectification Network for Fisheye Images](querycdr_query-based_controllable_distortion_rectification_network_for_fisheye_i.md)
- [\[ECCV 2024\] Optimizing Illuminant Estimation in Dual-Exposure HDR Imaging](optimizing_illuminant_estimation_in_dual-exposure_hdr_imaging.md)
- [\[ECCV 2024\] RAW-Adapter: Adapting Pre-trained Visual Model to Camera RAW Images](raw-adapter_adapting_pre-trained_visual_model_to_camera_raw_images.md)

</div>

<!-- RELATED:END -->
