---
title: >-
  [Paper Note] LUMINA: A Multi-Vendor Mammography Benchmark with Energy Harmonization Protocol
description: >-
  [CVPR 2026][Medical Imaging][mammography] This paper introduces LUMINA, a multi-vendor full-field digital mammography (FFDM) dataset comprising 468 patients and 1,824 images…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "mammography"
  - "multi-vendor dataset"
  - "energy harmonization"
  - "histogram matching"
  - "benchmark"
date: 2026-05-08
content_hash: 8090fb235833770f
---

# LUMINA: A Multi-Vendor Mammography Benchmark with Energy Harmonization Protocol

**Conference**: CVPR 2026
**arXiv**: [2603.14644](https://arxiv.org/abs/2603.14644)  
**Code**: [Available](https://github.com/NUBagciLab/LUMINA)  
**Area**: Medical Imaging
**Keywords**: mammography, multi-vendor dataset, energy harmonization, histogram matching, benchmark

## TL;DR

This paper introduces LUMINA, a multi-vendor full-field digital mammography (FFDM) dataset comprising 468 patients and 1,824 images, accompanied by a foreground-pixel histogram matching protocol for energy harmonization. The benchmark systematically evaluates CNN and Transformer models across three clinical tasks: diagnosis, BI-RADS classification, and breast density prediction.

## Background & Motivation

- **Background**: Existing public mammography datasets (e.g., CBIS-DDSM, INbreast) suffer from notable deficiencies in scale, clinical annotation completeness, and vendor diversity. CBIS-DDSM is derived from legacy screen-film mammography (SFM) scans, while INbreast contains only 115 patients.
- **Limitations of Prior Work**: Multi-vendor acquisition systems differ in energy settings (high/low energy) and vendor-specific processing pipelines, resulting in substantial domain shift in image appearance and intensity distributions. Consequently, models generalize poorly across vendors.
- **Goal**: (1) Construct an FFDM benchmark dataset that emphasizes vendor diversity and energy metadata; (2) propose a model-agnostic foreground histogram harmonization method to eliminate vendor/energy-induced domain shift.

## Method

### Overall Architecture

The LUMINA workflow consists of three stages: (1) **Data collection and curation** — 1,824 FFDM images from six vendors, with pathology-confirmed malignancy labels, BI-RADS scores, and breast density annotations; (2) **Foreground histogram harmonization (Energy Harmonization)** — aligning all images to a low-energy reference distribution; (3) **Multi-task benchmark evaluation** — comparing CNNs (ResNet-50, DenseNet-121, EfficientNet-B0) and a Transformer (Swin-T) across the three clinical tasks.

### Key Designs

1. **Multi-Vendor Dataset Construction**: Data were collected from six vendors — IMS, Metaltronica, FUJIFILM, Siemens, Carestream, and GE — covering 468 patients (250 benign, 218 malignant) in 12–14 bit DICOM format. Annotations include pathology-confirmed outcomes, BI-RADS grades 0–6, and breast density categories A–D. Images in FUJIFILM's MONOCHROME1 format are uniformly converted to MONOCHROME2.

2. **Foreground-Only CDF Matching**: The core idea is to exclude background pixels (intensity = 0) and perform CDF matching exclusively on the foreground breast region. Specifically, a foreground mask is defined as $M_s = \{(x,y) \mid \mathbf{I}_s(x,y) > 0\}$; foreground histograms $H_s(k)$ and $H_r(k)$ are computed for the source and reference images respectively, normalized into CDFs $\bar{C}_s(p)$ and $\bar{C}_r(q)$, and intensity transformation is achieved via the mapping $\mathcal{T}(p) = \arg\min_q |\bar{C}_s(p) - \bar{C}_r(q)|$. The reference histogram is drawn from the low-energy FFDM subset using 12-bit bins to preserve fine-grained detail. **Design Motivation**: Standard histogram matching is severely distorted by large areas of black background pixels; the foreground mask effectively mitigates this problem.

3. **Dual-View Shared-Backbone Network**: CC (craniocaudal) and MLO (mediolateral oblique) views are processed through a weight-sharing backbone independently; the resulting features are concatenated and passed to a fully connected classifier. Compared to independent backbones, weight sharing reduces parameters by 48% (4.34M vs. 8.34M) while achieving comparable or superior performance.

### Loss & Training

- Standard cross-entropy classification loss
- AdamW optimizer: $\text{lr}=1 \times 10^{-3}$ for CNNs; $\text{lr}=1 \times 10^{-5}$ for Swin-T
- 100 epochs with learning rate decay by 0.1 every 30 epochs; weight decay $1 \times 10^{-5}$
- 5-fold cross-validation; best model selected by validation AUC
- Data augmentation limited to horizontal flipping and resizing; grayscale images replicated to three channels
- PyTorch with CUDA determinism flags for reproducibility
- Training environment: 8 × NVIDIA A6000 GPUs

## Key Experimental Results

### Main Results

| Dataset / Task | Metric | Ours (Best) | Prev. SOTA | Notes |
|---|---|---|---|---|
| Diagnosis (Two-view, 512²) | AUC | **93.54%** (EfficientNet-B0) | — | Best overall: dual-view + high resolution |
| Diagnosis (Single-view, 512²) | AUC | 92.13% (EfficientNet-B0) | — | Second best among single-view configs |
| BI-RADS Binary (224²) | AUC | **92.80%** (EfficientNet-B0) | — | Low/high risk classification |
| BI-RADS Ternary (224²) | AUC | 83.27% (EfficientNet-B0) | — | Low/intermediate/high risk |
| Density Prediction (224²) | Macro-AUC | **89.43%** (Swin-T) | — | Transformer better suited for density |

### Ablation Study

| Configuration | Key Metric (AUC) | Notes |
|---|---|---|
| Shared backbone EfficientNet-B0 (224²) | 92.99% | 4.34M parameters |
| Independent backbone EfficientNet-B0 (224²) | 93.54% | 8.34M parameters; double the params, marginal gain |
| Raw images (no harmonization) | Baseline | Lower AUC across all tasks |
| Foreground histogram harmonization | +Gain | Consistent improvements in ACC/AUC/F1; Grad-CAM more focused |

### Key Findings

- Dual-view models consistently outperform single-view counterparts, confirming the complementary value of CC and MLO views.
- EfficientNet-B0 achieves the best performance on diagnosis and BI-RADS tasks with only ~4M parameters; Swin-T excels at density prediction.
- Higher input resolution (512²) generally improves performance, though 224² remains competitive with substantially lower computational cost.
- Histogram harmonization not only improves quantitative metrics but also sharpens Grad-CAM attention, directing model focus toward lesion regions.
- Low-energy images benefit most from harmonization, as high-energy images dominate the dataset distribution.

## Highlights & Insights

- **Practical value of foreground masking**: A simple yet effective idea — performing histogram matching after excluding background pixels. Though seemingly straightforward, this design is critical in mammography, where FFDM images contain large areas of black background.
- **Model-agnostic preprocessing**: The harmonization protocol can be applied as a lightweight preprocessing step to any backbone, making it straightforward to adopt in practice.
- **Dataset systematicity**: The combination of complete annotations (pathology + BI-RADS + density) with vendor/energy metadata is unique among existing datasets.
- **Clinical insight**: EfficientNet-B0 wins on diagnostic tasks with the fewest parameters, while Swin-T's global attention mechanism makes it better suited for density prediction — revealing a meaningful relationship between task type and model selection.

## Limitations & Future Work

- The dataset scale remains modest (468 patients) compared to large-scale resources such as EMBED (~500K images).
- Data originate from a single institution in Turkey, limiting patient population diversity.
- The reference distribution for harmonization is a representative low-energy FFDM subset; no adaptive reference selection mechanism is explored.
- More advanced domain adaptation methods (e.g., adversarial training, frequency-domain alignment) are not investigated.
- No direct experimental comparison with established cross-vendor harmonization methods (e.g., ComBat, HarmoFL) is provided.
- The four-view model underperforms the dual-view model, likely due to overfitting caused by excessive parameters on a small dataset.

## Related Work & Insights

- ComBat corrects batch effects via empirical Bayes in feature space rather than pixel space.
- HarmoFL reduces cross-site variation through frequency-domain amplitude normalization in federated learning settings.
- LUMINA complements datasets such as VinDr-Mammo (5,000 patients, single vendor, Vietnam) and RSNA (1,970 patients) — smaller in scale but offering greater vendor diversity.
- The pixel-space approach proposed here is more interpretable and requires no training, making it complementary to feature-space methods.
- **Insight**: For multi-center medical imaging research, lightweight pixel-space preprocessing may be more practical than complex domain adaptation pipelines.
- Combining LUMINA with MIL-PF (also CVPR 2026) is a promising direction: applying LUMINA's harmonization as preprocessing followed by frozen encoder + MIL classification.

## Rating

- **Novelty**: ⭐⭐⭐ — Solid dataset contribution, though the proposed method (foreground histogram matching) is technically straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers three tasks, multiple models, multiple resolutions, ablations, visualizations, and energy-level analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Rich tables and figures, transparent experimental setup, and a persuasive dataset comparison table.
- **Value**: ⭐⭐⭐⭐ — The multi-vendor benchmark makes a direct contribution to the community and is publicly released on OSF, Kaggle, and GitHub.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification](milpf_multiple_instance_learning_on_precomputed_fe.md)
- [\[CVPR 2026\] A protocol for evaluating robustness to H&E staining variation in computational pathology models](a_protocol_for_evaluating_robustness_to_he_stainin.md)
- [\[CVPR 2026\] MedGEN-Bench: Contextually Entangled Benchmark for Open-Ended Multimodal Medical Generation](medgen-bench_contextually_entangled_benchmark_for_open-ended_multimodal_medical_.md)
- [\[NeurIPS 2025\] Pancakes: Consistent Multi-Protocol Image Segmentation Across Biomedical Domains](../../NeurIPS2025/medical_imaging/pancakes_consistent_multi-protocol_image_segmentation_across_biomedical_domains.md)
- [\[ICCV 2025\] Progressive Test Time Energy Adaptation for Medical Image Segmentation](../../ICCV2025/medical_imaging/progressive_test_time_energy_adaptation_for_medical_image_segmentation.md)

</div>

<!-- RELATED:END -->
