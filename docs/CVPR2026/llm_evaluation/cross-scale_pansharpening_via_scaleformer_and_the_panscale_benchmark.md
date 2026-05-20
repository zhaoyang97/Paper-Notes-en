---
title: >-
  [Paper Note] Cross-Scale Pansharpening via ScaleFormer and the PanScale Benchmark
description: >-
  [CVPR 2026][LLM Evaluation][Remote sensing image fusion] This paper proposes PanScale, the first cross-scale pansharpening dataset, along with the PanScale-Bench evaluation benchmark…
tags:
  - "CVPR 2026"
  - "LLM Evaluation"
  - "Remote sensing image fusion"
  - "cross-scale generalization"
  - "Transformer"
  - "rotary position encoding"
  - "pansharpening"
date: 2026-05-08
content_hash: 329015a8241aad6e
---

# Cross-Scale Pansharpening via ScaleFormer and the PanScale Benchmark

**Conference**: CVPR 2026
**arXiv**: [2603.00543](https://arxiv.org/abs/2603.00543)  
**Code**: [GitHub](https://github.com/caoke-963/ScaleFormer)  
**Area**: LLM Evaluation
**Keywords**: Remote sensing image fusion, cross-scale generalization, Transformer, rotary position encoding, pansharpening

## TL;DR
This paper proposes PanScale, the first cross-scale pansharpening dataset, along with the PanScale-Bench evaluation benchmark, and the ScaleFormer framework — which reinterprets resolution variation as sequence length variation, achieving cross-scale generalization via Scale-Aware Patchify bucketed sampling, decoupled spatial-sequence modeling, and RoPE.

## Background & Motivation
**Background**: Pansharpening fuses high-resolution panchromatic (PAN) images with low-resolution multispectral (LRMS) images to produce high-resolution multispectral (HRMS) images, a core task in remote sensing image processing. CNN/Transformer-based methods (MSDCNN, HFIN, ARConv, etc.) have achieved substantial progress.

**Limitations of Prior Work**: (i) **Computational and memory bottlenecks** — scaling from training crop sizes (200–256 px) to inference at 800/1600/2000 px causes Transformer memory to explode, with standard GPUs frequently running out of memory at 800 px; (ii) **Patch inference artifacts** — forced patch-based inference introduces boundary discontinuities and visible block artifacts; (iii) **Weak cross-scale generalization** — training at a single low resolution induces scale-induced distribution shift, with brightness distributions shifting noticeably as resolution increases.

**Key Challenge**: Existing datasets (PanCollection, NBU, PAirMax) provide only limited scale diversity and resolution range, lacking a standardized multi-scale, high-resolution evaluation protocol.

**Goal**: Systematically address the challenges of cross-scale pansharpening across three dimensions: data, algorithm, and computation.

**Key Insight**: Reframe resolution variation as sequence length variation — spatial patch size is fixed, so only the sequence length grows linearly with image scale.

**Core Idea**: Introduce a sequence axis via Scale-Aware Patchify, decouple spatial modeling from scale modeling, and leverage RoPE to extrapolate to unseen scales.

## Method

### Overall Architecture
ScaleFormer consists of three core components:
1. **Scale-Aware Patchify (SAP)**: A bucketed window sampling strategy.
2. **Single Transformer module**: Spatial Transformer (spatial-domain modeling) + Sequence Transformer (sequence/scale-domain modeling).
3. **Cross Transformer module**: Spatial-Cross + Sequence-Cross Transformer for cross-modal feature fusion.

Given input PAN image $\mathbf{P} \in \mathbb{R}^{H \times W \times 1}$ and upsampled MS image $\mathbf{L} \in \mathbb{R}^{H \times W \times C}$, SAP converts them into 5D tensors $\mathbf{P}_{5d} \in \mathbb{R}^{B \times T \times C \times h \times w}$, where $T$ is the sequence length.

### Key Designs
1. **Scale-Aware Patchify (SAP)**: During training, a bucket index $t$ is randomly sampled to determine window size $w(t)$; a Patch-to-Sequence Tokenizer partitions the input into token sequences of varying lengths, exposing the model to a range of effective sequence lengths. At inference, a fixed window size is used, and higher resolutions are handled solely by extending the sequence. The key effect is to prevent mean and variance drift, stabilizing per-token statistics.

2. **Decoupled Spatial-Sequence Modeling**: The Spatial Transformer models intra-patch spatial relationships:
$$\mathbf{f}_{i,1} = \mathbf{f}_i + SA_{spa}(LN(\mathbf{f}_i))$$
The Sequence Transformer models cross-patch correlations along the sequence dimension:
$$\mathbf{f}_{i+1,1} = \mathbf{f}_{i+1} + SA_{seq}(LN(\mathbf{f}_{i+1}))$$
where $SA_{seq}$ merges the batch and spatial dimensions during operation, and injects **RoPE** to encode continuous relative position information, enhancing scale extrapolation capability.

3. **Cross Transformer module**: A similar architecture using cross-attention to enable PAN–MS cross-modal interaction:
$$\mathbf{f}_{i,1}^{ms} = \mathbf{f}_i^{ms} + CA_{spa}(LN(\mathbf{f}_i^{ms}), LN(\mathbf{f}^{pan}))$$

### Loss & Training
An L1 loss $\mathbf{L} = \|\mathbf{H}_{out} - \mathbf{G}\|_1$ is used. The model is trained with the Adam optimizer, initial learning rate $5 \times 10^{-4}$, cosine annealing decay to $5 \times 10^{-8}$, for 500 epochs on an NVIDIA 3090, with 32 channels.

## Key Experimental Results

### Main Results: Averaged Results Across Three PanScale Subsets

| Method | Jilin PSNR/SSIM | Landsat PSNR/SSIM | Skysat PSNR/SSIM |
|------|-----------------|-------------------|-------------------|
| HFIN | 38.00/0.9698 | 40.21/0.9666 | 43.96/0.9658 |
| ARConv | 38.23/0.9697 | 39.66/0.9638 | 43.40/0.9797 |
| Pan-mamba | 35.55/0.9480 | 36.73/0.9206 | 41.39/0.9493 |
| **ScaleFormer** | **39.29/0.9761** | **41.04/0.9711** | **44.65/0.9827** |

ScaleFormer outperforms all SOTA methods across all datasets, with stable performance as resolution increases.

### Ablation Study: Landsat Dataset

| Configuration | 200px PSNR | 400px PSNR | 800px PSNR | 1600px PSNR |
|---------|-----------|-----------|-----------|------------|
| w/o RoPE | 40.46 | 40.95 | 40.76 | 40.69 |
| SeqT→SpaT | 40.91 | 41.30 | 40.72 | 40.51 |
| w/o SAP | 40.53 | 40.93 | 40.62 | 40.39 |
| **Full Model** | **40.61** | **41.37** | **41.13** | **41.03** |

All ablated variants exhibit noticeable performance degradation at higher resolutions, confirming that each component is indispensable for cross-scale generalization.

### Key Findings
- ScaleFormer has only 0.52M parameters (1/4 of HFIN, 1/9 of ARConv), with a significant computational efficiency advantage.
- GFLOPs and memory usage of ScaleFormer grow substantially more slowly than those of HFIN/ARConv as resolution increases.
- ARConv exhibits severe block artifacts under patch-based inference (significant drop in DDC-IoU).
- ScaleFormer remains competitive in full-resolution real-world scene evaluation (without ground truth).

## Highlights & Insights
- **Elegant problem reformulation**: Recasting resolution generalization as sequence length generalization draws on sequence modeling ideas from NLP and video models.
- **Outstanding computational efficiency**: Substantially fewer parameters and GFLOPs than SOTA methods, with advantages widening at higher resolutions.
- **Dataset contribution**: PanScale is the first cross-scale pansharpening dataset covering three satellite platforms (0.5–15 m resolution).
- **Novel application of RoPE**: Adapts RoPE from text/video domains to remote sensing fusion tasks for scale extrapolation.

## Limitations & Future Work
- The approach focuses solely on pansharpening; generalization to other remote sensing fusion tasks (hyperspectral fusion, SAR–optical fusion) remains unvalidated.
- The bucketing strategy in SAP uses a predefined fixed set of window sizes; an adaptive strategy may be more effective.
- Only L1 loss is employed; perceptual losses or GAN losses may further improve visual quality.
- The self-attention in the Sequence Transformer remains $O(T^2)$, which may become a bottleneck for extremely large-scale inputs.

## Related Work & Insights
- Traditional methods (GS, IHS, GFPCA) perform poorly in cross-scale settings (PSNR over 10 dB lower).
- CNN-based methods (MSDCNN, SFINet, MSDDN) offer limited cross-scale generalization.
- HFIN/ARConv represent the current SOTA but suffer from severe memory and computational bottlenecks.
- Pan-mamba adopts the Mamba architecture but underperforms Transformer-based approaches.
- FlexViT's multi-resolution training and bucketed training strategies in video generation inspired the design of SAP.

## PanScale Dataset Details
- **Three sub-datasets**: Jilin (Jilin-1 satellite, 0.5–1 m resolution), Landsat (Landsat-8, 15 m resolution), and Skysat (Planet SkySat, ~1 m resolution).
- **Test set design**: Each sub-dataset includes both reduced-resolution (200×200 to 2000×2000) and full-resolution multi-scale test sets.
- **Data source**: Acquired and preprocessed via Google Earth Engine (GEE).
- **Evaluation metrics**: PanScale-Bench integrates reference metrics (PSNR/SSIM/ERGAS/Q) and no-reference metrics ($D_\lambda$/$D_S$/QNR).

## Efficiency Comparison

| Method | Parameters (M) | GFLOPs (G) |
|------|-----------|----------|
| ARConv | 4.4147 | 38.32 |
| HFIN | 1.9836 | 46.21 |
| **ScaleFormer** | **0.5151** | **20.57** |

## Rating ⭐
- **Novelty**: ⭐⭐⭐⭐ — The resolution-to-sequence-length reformulation is original; the SAP + RoPE combination is effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive coverage across three datasets, multiple scales, full-resolution evaluation, ablation studies, efficiency analysis, and visualization.
- **Writing Quality**: ⭐⭐⭐⭐ — Excellent figure and table design; Fig. 1/2 clearly illustrate the problem and compare solutions.
- **Value**: ⭐⭐⭐⭐⭐ — A unified contribution of dataset, benchmark, and method, advancing the remote sensing fusion field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation](../../NeurIPS2025/llm_evaluation/parrot_a_benchmark_for_evaluating_llms_in_cross-system_sql_translation.md)
- [\[CVPR 2026\] SATTC: Structure-Aware Label-Free Test-Time Calibration for Cross-Subject EEG-to-Image Retrieval](sattc_structure-aware_label-free_test-time_calibration_for_cross-subject_eeg-to-.md)
- [\[CVPR 2026\] HyCal: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning](hycal_training_free_prototype_calibration_for_cross_discipline_fscil.md)
- [\[CVPR 2026\] TacSIm: A Dataset and Benchmark for Football Tactical Style Imitation](tacsim_a_dataset_and_benchmark_for_football_tactical_style_imitation.md)
- [\[ACL 2026\] TingIS: Real-time Risk Event Discovery from Noisy Customer Incidents at Enterprise Scale](../../ACL2026/llm_evaluation/tingis_real-time_risk_event_discovery_from_noisy_customer_incidents_at_enterpris.md)

</div>

<!-- RELATED:END -->
