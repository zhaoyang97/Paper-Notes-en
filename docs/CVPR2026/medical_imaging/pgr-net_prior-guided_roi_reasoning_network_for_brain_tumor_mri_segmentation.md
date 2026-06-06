---
title: >-
  [Paper Note] PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation
description: >-
  [CVPR 2026][Medical Imaging][Brain tumor segmentation] PGR-Net proposes an explicit ROI-aware brain tumor MRI segmentation network that concentrates computational resources on lesion regions via a data-driven spatial pri…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Brain tumor segmentation"
  - "ROI prior"
  - "spatial guidance"
  - "RetNet"
  - "MRI"
date: 2026-05-08
content_hash: 8e14e9ef4c4f857a
---

# PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.21626](https://arxiv.org/abs/2603.21626)  
**Code**: [https://github.com/CNU-MedAI-Lab/PGR-Net](https://github.com/CNU-MedAI-Lab/PGR-Net)  
**Area**: Medical Image Segmentation
**Keywords**: Brain tumor segmentation, ROI prior, spatial guidance, RetNet, MRI

## TL;DR

PGR-Net proposes an explicit ROI-aware brain tumor MRI segmentation network that concentrates computational resources on lesion regions via a data-driven spatial prior template set $\{(r_i, c_i)\}$ constructed from the training set, a hierarchical Top-K ROI selection mechanism, and a Window Gaussian-Spatial decay guidance module (WinGS-ROI). With only 8.64M parameters, the method achieves state-of-the-art performance on BraTS-2019/2023 and MSD Task01.

## Background & Motivation

**Background**: Brain tumor MRI segmentation is a foundational task for clinical diagnosis and radiotherapy target delineation. Segmentation accuracy has steadily improved, progressing from UNet to TransUNet, Swin UNETR, and SSM-based methods such as Mamba-UNet.

**Limitations of Prior Work**: Brain tumors exhibit severe **spatial sparsity** in MRI — the average tumor region in BraTS2023 accounts for only approximately 10.7% of the full image (roughly 2,740 pixels in a 160×160 slice). This causes models to be dominated by background features in early training; even after roughly localizing the tumor, a substantial proportion of computation is still spent on healthy tissue. Existing models generally assume a uniform lesion distribution and overlook the clinically well-established spatial distribution patterns of brain tumors.

**Key Challenge**: Tumors in the brain follow recognizable spatial distribution patterns — centers are predominantly concentrated at the frontal-temporal junction, while occipital occurrences are rare. Existing segmentation networks ignore these priors and allocate computation uniformly across the entire image, representing a significant waste. The few methods that introduce hard ROI guidance suffer from poor generalizability due to their inability to capture distributional patterns.

**Goal**: (1) Model tumor location and scale priors from data statistics; (2) leverage these priors for progressive hierarchical ROI selection; (3) embed learnable spatial guidance at each network layer to focus on lesions and suppress background.

**Key Insight**: The authors observe that the spatial distribution of brain tumors exhibits statistical regularity (evidenced by analyzing lesion center and scale distributions in the training set). A ROI prior template set $\{(r_i, c_i)\}$ can therefore be extracted from the training set and explicitly injected into the network to direct computation toward regions of interest.

**Core Idea**: Construct a data-driven tumor spatial prior template set, and realize globally-to-locally ROI-aware segmentation within a RetNet backbone through hierarchical Top-K ROI selection combined with Window Gaussian-Spatial decay guidance maps.

## Method

### Overall Architecture

PGR-Net adopts an encoder-decoder architecture with a windowed RetNet (Win-RetNet) backbone. The core pipeline proceeds as follows: (1) A ROI prior template set $\{(r_i, c_i)\}_{i=1}^N$ is constructed from the training set, encoding representative lesion scale ratios and center coordinates. (2) During encoding, the Hierarchical Top-K ROI Decision (HTK) module progressively filters high-confidence ROI candidates from the coarsest layer down, ultimately locking onto the optimal ROI. (3) The WinGS-ROI module generates center-enhanced Gaussian guidance maps for each candidate ROI and embeds them into the encoder, skip connections, and upsampling layers. (4) The decoding stage employs ROI-Only upsampling and ROI-Aware skip connections, operating exclusively within the identified ROI.

### Key Designs

1. **ROI Prior Template Construction**:

    - **Function**: Extract representative distributions of tumor location and scale from training set statistics.
    - **Mechanism**: Connected components are extracted from all training masks; the maximum side length $s$ of the minimum bounding rectangle and the center coordinate of each component are computed. After filtering out excessively small regions, the scale distribution is analyzed for local maxima (peaks). The top-$N$ peaks are taken as candidate scales; for each scale, the corresponding center coordinates are clustered and averaged, yielding $N$ prior templates $(r_i, c_i)$. A spatial clustering constraint with neighborhood radius $d=30$ prevents merging of same-scale regions that are spatially distant.
    - **Design Motivation**: Unlike hard-coded fixed ROIs, data-driven priors adapt to the distributional characteristics of different datasets and cover multi-scale candidates, providing sufficient search space for subsequent hierarchical selection.

2. **Hierarchical Top-K ROI Decision (HTK)**:

    - **Function**: Progressively filter high-confidence ROI candidates to ultimately localize the most precise lesion region.
    - **Mechanism**: At the coarsest encoding layer $l=L$, all $N$ ROI candidates are scored by a lightweight MLP and the top $K^{(L)}$ are retained. At finer layers $l < L$, only the candidates preserved from the previous layer are re-scored and further filtered. Cross-layer scores are normalized via softmax and aggregated into a global confidence matrix $S = \sum_l \alpha_l \hat{s}^{(l)}$, and the final selection is $R^* = \arg\max_i S_i$. A confidence gap $\Delta_{gap}$ and information entropy $H$ serve as stability criteria: if $\Delta_{gap} < \tau_1$ or $H > \tau_2$, the model falls back to full-image mode to prevent mislocalization.
    - **Design Motivation**: Coarse-to-fine hierarchical filtering is more robust than single-layer decision-making — coarse layers have large receptive fields suited to global localization, while fine layers have high spatial resolution suited to precise localization. The fallback mechanism ensures that abnormal samples (morphologically atypical or distribution-shifted) do not lead to catastrophic errors. The empirically observed fallback rate ranges between 3.5% and 7%.

3. **WinGS-ROI Window Gaussian-Spatial Decay Guidance**:

    - **Function**: Generate spatially guided maps with center enhancement and boundary smooth decay.
    - **Mechanism**: Each ROI candidate is modeled as a circular Gaussian template $G_i^{(l)}(u,v) = \rho_i \exp\!\left(-\frac{(u-x_i)^2+(v-y_i)^2}{2\sigma_i^2}\right)$, weighted by the HTK confidence score $\rho_i$. Outside the ROI, a radial spatial decay $\exp\!\left(-\frac{(d_i-R_i)^2}{2\tau^2}\right)$ is applied. Templates from all ROI candidates are aggregated into a guidance map $M^{(l)}$, which modulates features multiplicatively as $\tilde{F}^{(l)} = (1 + \lambda M^{(l)}) \odot F^{(l)}$ to enhance lesion-region features. Upon high-confidence ROI locking, the module switches to a hard circular mask. WinGS-ROI is embedded at three locations: within the encoder (Win-RetNet), in the skip connections (ROI-Aware), and in the upsampling stage (ROI-Only).
    - **Design Motivation**: Center enhancement combined with boundary decay makes the network most sensitive to the lesion center while maintaining structural continuity rather than hard truncation. Multiplicative modulation preserves gradient flow from the original features without completely suppressing non-ROI information — except when a high-confidence lock is engaged.

### Loss & Training

The loss is a weighted combination of Dice loss and BCE loss (ratio 2:8). The Adam optimizer is used with an initial learning rate of 1e-3, trained for 300 epochs with 50-epoch early stopping. All experiments are independently run 3 times and averaged. HTK and segmentation losses are trained end-to-end without requiring additional ROI localization annotations.

## Key Experimental Results

### Main Results

BraTS-2023 Dice (%) comparison:

| Method | Params | Dice_WT | Dice_TC | Dice_ET | HD95_WT |
|--------|--------|---------|---------|---------|---------|
| UNet | 39.40M | 90.71 | 93.05 | 93.36 | 1.1863 |
| Swin UNETR | 25.11M | 91.11 | 93.20 | 93.42 | 1.1629 |
| Mamba-UNet | 35.86M | 91.03 | 93.32 | 93.31 | 1.1734 |
| M-Net | 81.59M | 91.33 | 93.55 | 93.42 | 1.1534 |
| VM-UNet | 44.28M | 90.52 | 93.40 | 93.50 | 1.1806 |
| **PGR-Net** | **8.64M** | **91.82** | **94.07** | **93.88** | **1.1334** |

Computational efficiency comparison:

| Method | Params (M) | FLOPs (G) | Inference Time |
|--------|-----------|----------|----------------|
| UNet | 39.40 | 321.19 | 12:32 |
| Swin UNETR | 25.11 | 106.80 | 21:33 |
| M-Net | 81.59 | 91.29 | 15:33 |
| **PGR-Net** | **8.64** | **39.05** | **9:41** |

### Ablation Study

BraTS-2019 / BraTS-2023 Dice (%) ablation:

| Configuration | Dice_WT | Dice_TC | Dice_ET |
|---------------|---------|---------|---------|
| Baseline (no modules) | 87.82 / 91.06 | 88.91 / 92.97 | 91.05 / 93.13 |
| + ROI Win-RetNet | 87.85 / 91.10 | 88.89 / 93.02 | 91.15 / 93.08 |
| + HTK | 88.55 / 91.66 | 89.64 / 93.42 | 91.99 / 93.35 |
| + WinGS-ROI (encoder) | 88.63 / 91.76 | 90.33 / 93.75 | 92.72 / 93.57 |
| + WinGS-ROI (skip connections) | 88.85 / 91.80 | 90.32 / 93.79 | 92.88 / 93.74 |
| + WinGS-ROI (upsampling) — full | **89.02 / 91.82** | **90.69 / 94.07** | **93.61 / 93.88** |

### Key Findings

- **HTK contributes the largest single-step improvement**: Adding HTK yields WT Dice gains of 0.70/0.56 and TC gains of 0.75/0.40, demonstrating that hierarchical ROI selection effectively focuses computation on the lesion region.
- WinGS-ROI contributes most in the encoder (particularly for ET, which improves from 91.99 to 92.72), with further cumulative gains in skip connections and upsampling.
- PGR-Net uses only 8.64M parameters (4.6× fewer than UNet, 9.4× fewer than M-Net) and 39.05G FLOPs, achieving the fastest inference (9:41 vs. 12–30+ minutes for other methods).
- The full-image fallback mode is triggered on only 3.52% of samples in BraTS-2023, confirming that prior-guided localization is reliable in the vast majority of cases.
- PGR-Net consistently outperforms all comparison methods across three datasets, with the most pronounced improvement in the WT region (the prior is primarily constructed from WT).

## Highlights & Insights

- **"Spend resources where they matter" design philosophy**: Brain tumors occupy less than 11% of the image; PGR-Net leverages ROI priors to concentrate computation on this 11%, achieving the best performance with the fewest parameters and FLOPs. This paradigm generalizes to all spatially sparse segmentation tasks (e.g., pulmonary nodules, small organ segmentation, retinal lesion detection).
- The combination of **data-driven priors and hierarchical decision-making** is noteworthy: the prior constrains the initial search space, while HTK dynamically refines localization at inference time — balancing the stability of priors with inference-time flexibility.
- **WinGS-ROI's soft design outperforms hard masking**: Gaussian center enhancement ensures maximal feature modulation at the lesion center, while boundary decay avoids artifacts from hard truncation. Hard masking is only applied after high-confidence locking.

## Limitations & Future Work

- ROI priors are constructed solely from the WT (Whole Tumor) region; independent priors for TC and ET are not built — multi-region priors may further improve small-region segmentation.
- All experiments are conducted on 2D slices (due to GPU constraints), foregoing the contextual information available in 3D volumes — a 3D extension is expected to yield further improvements.
- Since priors are constructed from the training set, their robustness to distribution shift (e.g., different hospitals or scanners) remains to be validated.
- Generalizability to other sparse segmentation tasks beyond brain tumors has not been verified.
- Although the RetNet backbone is efficient, its unidirectional sequence modeling may be limited in regions requiring bidirectional context (e.g., symmetric structures).

## Related Work & Insights

- **vs. nnUNet**: nnUNet is a classic automated segmentation method; PGR-Net improves WT Dice from 90.34 to 91.82 on BraTS-2023 while dramatically reducing inference time from 86:52 to 9:41.
- **vs. Swin UNETR**: PGR-Net uses 2.9× fewer parameters (8.64M vs. 25.11M) and achieves 0.71 higher WT Dice.
- **vs. Mamba-UNet**: SSM-based Mamba methods offer efficiency advantages but are less accurate than PGR-Net (91.03 vs. 91.82 WT Dice) and have 4× more parameters.
- **vs. MedSAM**: Even the foundation model MedSAM (240M parameters) underperforms PGR-Net on brain tumor segmentation, suggesting that domain-specific priors are more impactful than general-purpose large models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of data-driven priors, hierarchical Top-K ROI selection, and WinGS-ROI is novel; the approach of translating clinical observations into algorithmic design is inspiring.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets, three repeated runs, complete ablation (6 configurations with progressive addition), efficiency comparisons, and qualitative visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ The logic is clear; the progression from clinical observation to prior construction to network design is natural.
- **Value**: ⭐⭐⭐⭐ The approach of achieving SOTA with minimal parameters has practical value for the medical image segmentation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] M-Net: MRI Brain Tumor Sequential Segmentation Network via Mesh-Cast](../../ICCV2025/medical_imaging/m-net_mri_brain_tumor_sequential_segmentation_network_via_mesh-cast.md)
- [\[CVPR 2026\] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)
- [\[CVPR 2026\] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)
- [\[CVPR 2026\] Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)
- [\[CVPR 2026\] Virtual Full-stack Scanning of Brain MRI via Imputing Any Quantised Code](virtual_full-stack_scanning_of_brain_mri_via_imputing_any_quantised_code.md)

</div>

<!-- RELATED:END -->
