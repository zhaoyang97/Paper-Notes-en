---
title: >-
  [Paper Note] Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback
description: >-
  [CVPR2026][Human Understanding][Virtual try-on] This paper presents VTON-IQA, a reference-free image quality assessment framework for virtual try-on. It introduces VTON-QBench, a large-scale benchmark comprising 62,688 try-on images annotated with 431,800 human judgments, and proposes an Interleaved Cross-Attention (ICA) module to model interactions among garment, person, and try-on images, achieving image-level quality predictions that are closely aligned with human perception.
tags:
  - CVPR2026
  - Human Understanding
  - Virtual try-on
  - image quality assessment
  - reference-free evaluation
  - human feedback alignment
  - cross-attention
  - large-scale annotation benchmark
date: 2026-05-08
content_hash: 97f697c44dcd5db4
---

# Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback

**Conference**: CVPR2026
**arXiv**: [2603.13057](https://arxiv.org/abs/2603.13057)
**Code**: [GitHub](https://github.com/litelightlite/VTON-IQA)
**Area**: Human-Centric Understanding / Virtual Try-On Quality Assessment
**Keywords**: Virtual try-on, image quality assessment, reference-free evaluation, human feedback alignment, cross-attention, large-scale annotation benchmark

## TL;DR

This paper presents VTON-IQA, a reference-free image quality assessment framework for virtual try-on. It introduces VTON-QBench, a large-scale benchmark comprising 62,688 try-on images annotated with 431,800 human judgments, and proposes an Interleaved Cross-Attention (ICA) module to model interactions among garment, person, and try-on images, achieving image-level quality predictions that are closely aligned with human perception.

## Background & Motivation

1. **Absence of ground-truth references in real-world deployment**: In practical e-commerce settings, ground-truth images of the same person wearing the target garment are typically unavailable, rendering full-reference metrics such as SSIM and LPIPS inapplicable.
2. **Distribution-level metrics fail to reflect per-image quality**: FID and KID measure dataset-level statistical similarity and cannot assess the perceptual quality of individual generated images.
3. **Existing VTON evaluation methods lack large-scale human validation**: VTON-VLLM focuses on textual critique rather than quantitative scoring; VTBench employs LLM-based judgment without learning from large-scale human annotations; VTONQA is limited in scale (only 748 pairs annotated by 40 workers).
4. **Lack of publicly reproducible evaluation benchmarks**: Existing methods have not released implementations or standardized benchmarks, hindering reproducible evaluation.
5. **Try-on quality assessment differs from single-image IQA**: It requires simultaneous verification of garment fidelity and person identity preservation, necessitating cross-image interaction modeling by nature.
6. **Traditional metrics over-penalize global transformations**: SSIM and LPIPS are sensitive to pose and scale variations, leading to misalignment with human perception.

## Method

### Overall Architecture

VTON-IQA adopts a three-branch Transformer architecture that takes a garment image $I_G$, a person image $I_P$, and a generated try-on image $I_V$ as inputs, and produces a continuous quality score $\hat{s} \in [-1, 1]$. The backbone is DINOv3 ViT-L/16.

**Pipeline**: Each of the three images undergoes patch embedding with a [CLS] token → the first $L/2$ layers apply independent self-attention for feature extraction → the latter $L/2$ layers introduce ICA modules for cross-image interaction → [CLS] representations are extracted → weighted cosine similarity scoring is performed.

### Interleaved Cross-Attention (ICA) Module

The core design of ICA is **asymmetric interaction**: cross-attention layers are inserted between the self-attention and MLP components of standard Transformer blocks.

- The **try-on branch** aggregates information from both the garment and person branches: $\hat{X}_V = \tilde{X}_V + C_{V \leftarrow G} + C_{V \leftarrow P}$
- The **garment/person branches** receive information only from the try-on branch: $\hat{X}_G = \tilde{X}_G + C_{G \leftarrow V}$, $\hat{X}_P = \tilde{X}_P + C_{P \leftarrow V}$
- Bidirectional interactions between $(V, G)$ and $(V, P)$ are explicitly modeled, while **direct $G \leftrightarrow P$ coupling is deliberately avoided**, since quality judgment is inherently try-on-centric.

### Scoring Module

The [CLS] tokens $c_G, c_P, c_V$ from the three branches are extracted, and two cosine similarity terms are combined via a learnable weight $\alpha$:

$$\tilde{s} = \alpha \cdot \cos(c_G, c_V) + (1-\alpha) \cdot \cos(c_P, c_V)$$

The final score is mapped to $[-1, 1]$ via a learnable affine transformation followed by a tanh activation.

### Loss & Training

Two objectives are jointly optimized:

1. **Bradley–Terry preference learning**: Pairwise preferences between two try-on results generated for the same person–garment pair are modeled, and soft-label cross-entropy is used to align predicted preferences with human judgments.
2. **Score regression**: An L2 loss constrains the consistency between predicted scores and human ratings.

$$\mathcal{L} = -q_{ij}\log p_\theta - (1-q_{ij})\log(1-p_\theta) + \sum_{k}\|\Psi_\theta - S_k\|_2^2$$

### VTON-QBench Dataset Construction

| Dimension | Scale |
|-----------|-------|
| Garment–person pairs | 13,153 (with synthetic augmentation at 1.9×) |
| Try-on images | 62,688 |
| VTON models | 14 (covering GAN / U-Net Diffusion / DiT / commercial models) |
| Qualified annotators | 13,838 |
| Quality annotations | 431,800 |

**Three-level annotation**: Unnatural (1) / Slightly unnatural but not obvious (2) / Completely natural (3); the final score is the mean across multiple annotators.

**Data cleaning**: A two-stage filtering process is applied — (1) dummy question screening and anomalous behavior detection; (2) complete questionnaire removal when Krippendorff's $\alpha \leq 0.4$ — raising $\alpha$ from 0.286 to 0.550.

## Key Experimental Results

### Main Results: Comparison with Baselines (VTON-QBench Test Set)

| Method | SRCC↑ | PLCC↑ | R²↑ | A_macro↑ | A_micro↑ |
|--------|-------|-------|-----|----------|----------|
| SSIM | – | 0.135 | – | 0.596 | 0.593 |
| LPIPS | – | 0.387 | – | 0.701 | 0.695 |
| DINOv3 (zero-shot) | – | 0.261 | – | 0.637 | 0.641 |
| VTON-IQA w/o ICA | 0.617 | 0.615 | 0.372 | 0.722 | 0.747 |
| **VTON-IQA (full)** | **0.750** | **0.751** | **0.553** | **0.781** | **0.790** |

- The ICA module yields substantial gains of SRCC +0.133 and PLCC +0.136.
- Pairwise accuracy approaches human-level performance (human A_macro = 0.782 vs. model 0.771).

### Benchmark across 14 VTON Models (VITON-HD, Unpaired)

| Model | VTON-IQA↑ | FID↓ |
|-------|-----------|------|
| Nano Banana Pro | **0.315** | 10.309 |
| GPT-Image-1.5 | 0.234 | 12.801 |
| FitDit | 0.189 | 9.893 |
| Qwen-Image-Edit | 0.087 | 10.706 |
| IDM-VTON | 0.039 | 9.093 |
| OOTDiffusion | -0.142 | 9.064 |
| LADI-VTON | -0.864 | 21.515 |

Commercial models lead substantially on human-aligned scores; FID/KID rankings do not consistently correspond to human perception.

### Ablation Study

- **ICA vs. w/o ICA**: The ICA module yields significant improvements across all metrics, validating the necessity of cross-image interaction modeling.
- **Asymmetric vs. symmetric interaction**: The asymmetric design (avoiding $G \leftrightarrow P$ coupling) better reflects the semantic structure of try-on quality assessment.
- **Task-specific training vs. zero-shot**: Fine-tuning on VTON-QBench yields PLCC +0.354 over DINOv3 zero-shot, demonstrating the critical importance of in-domain training.

## Highlights & Insights

1. **Unprecedented dataset scale**: VTON-QBench is, to the best of the authors' knowledge, the largest human subjective evaluation dataset for virtual try-on, with a planned open-source release.
2. **Elegant asymmetric ICA design**: The try-on-centric interaction structure aligns with the semantics of quality assessment and avoids irrelevant coupling.
3. **Pairwise accuracy reaches human level**: The model's A_macro falls only 0.011 short of human performance, indicating practical utility in preference ranking.
4. **First unified benchmark across 14 VTON models**: Covering GAN, UNet-Diffusion, DiT, and commercial models, the results reveal a systematic discrepancy between traditional metrics and human perception.
5. **Complete synthetic data augmentation pipeline**: LoRA + FLUX.1-dev generation, GPT-based filtering, and human review expand garment–person pairs by 1.9×.

## Limitations & Future Work

1. **Coarse three-level annotation granularity**: Only three levels may fail to capture fine-grained quality differences; continuous ratings or multi-dimensional scoring may be preferable.
2. **Correlation metrics still lag behind humans**: SRCC = 0.750 vs. human 0.760; R² = 0.489 vs. 0.536, leaving room for improvement in absolute score prediction.
3. **Single overall score without sub-dimension diagnosis**: The framework outputs only a holistic quality score, lacking diagnostic capability for sub-dimensions such as garment texture, color, shape, and length.
4. **Heavy backbone**: Three-branch inference with DINOv3 ViT-L/16 incurs substantial computational cost, requiring efficiency considerations for deployment.
5. **Synthetic augmentation relies on a commercial model**: The pseudo-triplet construction uses Nano Banana Pro, limiting dataset construction cost and reproducibility.
6. **Dynamic scenarios such as video try-on remain unexplored** in terms of quality assessment.

## Related Work & Insights

| Method | Data Scale | Reference-Free | Image-Level Score | Human Annotation | Open-Source |
|--------|------------|----------------|-------------------|------------------|-------------|
| SSIM/LPIPS | N/A | ✗ | ✓ | ✗ | ✓ |
| FID/KID | N/A | ✓ | ✗ (distribution-level) | ✗ | ✓ |
| VTONQA | 748 pairs / 8,132 images / 40 workers | ✓ | ✓ | ✓ | ✗ |
| VTON-VLLM | – | ✓ | ✗ (textual) | ✓ | ✗ |
| VTBench | – | ✓ | ✓ | Indirect | – |
| **VTON-IQA** | **13K pairs / 63K images / 14K workers** | **✓** | **✓** | **✓** | **✓** |

VTON-IQA comprehensively surpasses prior work in data scale, evaluation completeness, and open-source commitment.

## Rating

- Novelty: ⭐⭐⭐⭐ — The asymmetric cross-image interaction design of ICA is original, and the large-scale VTON human annotation benchmark fills a notable gap in the field.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Benchmark across 14 models, human-level comparison, ablation studies, and qualitative analysis constitute an exceptionally comprehensive evaluation.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured presentation with detailed dataset construction descriptions and rigorous mathematical formulations.
- Value: ⭐⭐⭐⭐ — Provides the virtual try-on community with a standardized evaluation benchmark and tool, offering both engineering and academic value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RefTon: Reference Person Shot Assist Virtual Try-on](refton_reference_person_shot_assist_virtual_try-on.md)
- [\[CVPR 2026\] Mobile-VTON: High-Fidelity On-Device Virtual Try-On](mobile_vton_ondevice_virtual_tryon.md)
- [\[CVPR 2026\] rPPG-VQA: A Video Quality Assessment Framework for Unsupervised rPPG Training](rppg_vqa_video_quality_assessment.md)
- [\[CVPR 2026\] LASER: Layer-wise Scale Alignment for Training-Free Streaming 4D Reconstruction](laser_layer-wise_scale_alignment_for_training-free_streaming_4d_reconstruction.md)
- [\[CVPR 2026\] COG: Confidence-aware Optimal Geometric Correspondence for Unsupervised Single-reference Novel Object Pose Estimation](cog_confidence-aware_optimal_geometric_correspondence_for_unsupervised_single-re.md)

</div>

<!-- RELATED:END -->
